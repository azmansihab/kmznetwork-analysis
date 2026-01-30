import streamlit as st
import geopandas as gpd
import pandas as pd
import osmnx as ox
import networkx as nx
import simplekml
import zipfile
import numpy as np
import os
from scipy.spatial import ConvexHull, distance_matrix
from geopy.geocoders import ArcGIS
from concurrent.futures import ThreadPoolExecutor
from shapely.geometry import Point

# Pengaturan OSMNX v2.0+
ox.settings.use_cache = True
ox.settings.log_console = False
geolocator = ArcGIS(user_agent="isp_frontage_optimizer")

# --- Fungsi Pendukung ---
def extract_kml_from_kmz(kmz_file):
    try:
        with zipfile.ZipFile(kmz_file, 'r') as z:
            kml_filename = [f for f in z.namelist() if f.endswith('.kml')][0]
            with z.open(kml_filename) as f: return f.read()
    except: return None

def load_data(uploaded_file):
    try:
        if uploaded_file.name.lower().endswith('.kmz'):
            kml_data = extract_kml_from_kmz(uploaded_file)
            if kml_data:
                temp_path = f"temp_{uploaded_file.name}.kml"
                with open(temp_path, "wb") as f: f.write(kml_data)
                gdf = gpd.read_file(temp_path, driver='KML')
                if os.path.exists(temp_path): os.remove(temp_path)
                return gdf
        return gpd.read_file(uploaded_file, driver='KML')
    except Exception as e:
        st.error(f"Error: {e}")
        return None

@st.cache_data(show_spinner=False)
def get_street_name(lat, lon):
    try:
        location = geolocator.reverse(f"{lat}, {lon}", timeout=3)
        return location.raw['Address'].split(',')[0] if location else "Jalan Lokal"
    except: return "Jalan Lokal"

def batch_geocoding(coords_list):
    with ThreadPoolExecutor(max_workers=10) as executor:
        return list(executor.map(lambda p: get_street_name(p[0], p[1]), coords_list))

# --- UI Streamlit ---
st.set_page_config(page_title="ISP Planner - Frontage Optimized", layout="wide")
st.title("🌐 ISP Network Planner: Front-Facing Optimization")
st.markdown("Logika: Penarikan kabel diprioritaskan dari **depan rumah** (sisi jalan).")

if 'result_data' not in st.session_state:
    st.session_state.result_data = None

st.sidebar.header("⚙️ Konfigurasi")
max_dist = st.sidebar.slider("Radius Maksimal (Meter)", 50, 1000, 250)
max_homes = st.sidebar.slider("Kapasitas Rumah per Tiang", 1, 32, 10)

c1, c2 = st.columns(2)
with c1: file_tiang = st.file_uploader("Upload Tiang", type=['kml', 'kmz'])
with c2: file_rumah = st.file_uploader("Upload Rumah", type=['kml', 'kmz'])

if file_tiang and file_rumah:
    gdf_tiang = load_data(file_tiang)
    gdf_rumah = load_data(file_rumah)

    if gdf_tiang is not None and gdf_rumah is not None:
        st.success(f"✅ Terdeteksi: {len(gdf_tiang)} Tiang & {len(gdf_rumah)} Rumah")
        
        if st.button("🚀 JALANKAN ANALISIS (FRONT-FACING)"):
            with st.spinner("Menganalisis orientasi rumah terhadap jalan..."):
                avg_lat, avg_lon = gdf_tiang.geometry.y.mean(), gdf_tiang.geometry.x.mean()
                G = ox.graph_from_point((avg_lat, avg_lon), dist=3000, network_type='all')
                
                tiang_pts = np.array([[t.x, t.y] for t in gdf_tiang.geometry])
                rumah_pts = np.array([[r.x, r.y] for r in gdf_rumah.geometry])
                dist_mat = distance_matrix(rumah_pts, tiang_pts)
                
                final_alloc = []
                taken = set()
                caps = {i: 0 for i in range(len(tiang_pts))}
                
                # Optimasi Global
                for _ in range(len(rumah_pts)):
                    if np.isnan(np.nanmin(dist_mat)): break
                    r_idx, t_idx = np.unravel_index(np.nanargmin(dist_mat), dist_mat.shape)
                    
                    if r_idx not in taken and caps[t_idx] < max_homes:
                        # Logic: Find Frontage Node
                        r_node = ox.distance.nearest_nodes(G, rumah_pts[r_idx][0], rumah_pts[r_idx][1])
                        t_node = ox.distance.nearest_nodes(G, tiang_pts[t_idx][0], tiang_pts[t_idx][1])
                        
                        try:
                            dist = nx.shortest_path_length(G, t_node, r_node, weight='length')
                            if dist <= max_dist:
                                path = nx.shortest_path(G, t_node, r_node, weight='length')
                                p_coords = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in path]
                                # Add House Point as final tail of the line
                                p_coords.append((rumah_pts[r_idx][0], rumah_pts[r_idx][1]))
                                
                                final_alloc.append({
                                    't_name': gdf_tiang.iloc[t_idx].get('Name', f"T-{t_idx}"),
                                    'r_name': gdf_rumah.iloc[r_idx].get('Name', f"R-{r_idx}"),
                                    'lat': rumah_pts[r_idx][1], 'lon': rumah_pts[r_idx][0],
                                    'path': p_coords, 'dist': dist, 't_idx': t_idx,
                                    't_coord': (tiang_pts[t_idx][0], tiang_pts[t_idx][1])
                                })
                                taken.add(r_idx)
                                caps[t_idx] += 1
                        except: pass
                    dist_mat[r_idx, :] = np.nan

                # Batch Geocoding
                geo_req = [(a['lat'], a['lon']) for a in final_alloc]
                streets = batch_geocoding(geo_req)
                
                st.session_state.result_data = {
                    'alloc': final_alloc, 'streets': streets,
                    'uncovered': [gdf_rumah.iloc[j] for j in range(len(gdf_rumah)) if j not in taken]
                }

# Render Results
if st.session_state.result_data:
    data = st.session_state.result_data
    kml = simplekml.Kml()
    colors = [simplekml.Color.red, simplekml.Color.blue, simplekml.Color.green, simplekml.Color.yellow, simplekml.Color.purple]
    csv_list = []
    fols = {}
    bound_pts = {}

    for idx, a in enumerate(data['alloc']):
        s_name = data['streets'][idx]
        csv_list.append({
            'NO_RUMAH': a['r_name'], 'TIANG_TERHUBUNG': a['t_name'], 'NAMA_JALAN': s_name,
            'JARAK_KABEL_M': round(a['dist'], 2), 'LATITUDE': a['lat'], 'LONGITUDE': a['lon']
        })
        
        if a['t_name'] not in fols:
            fols[a['t_name']] = kml.newfolder(name=f"AREA_{a['t_name']}")
            fols[a['t_name']].newpoint(name=f"PUSAT_{a['t_name']}", coords=[a['t_coord']])
            bound_pts[a['t_name']] = [a['t_coord']]
            
        bound_pts[a['t_name']].append((a['lon'], a['lat']))
        p = fols[a['t_name']].newpoint(name=a['r_name'], coords=[(a['lon'], a['lat'])])
        p.description = f"Jalan: {s_name}\nJarak: {int(a['dist'])}m"
        ls = fols[a['t_name']].newlinestring(name=f"Drop {a['r_name']}", coords=a['path'])
        ls.style.linestyle.color = colors[a['t_idx'] % len(colors)]
        ls.style.linestyle.width = 3

    # Add Boundary Polygons
    for t_name, pts in bound_pts.items():
        if len(pts) >= 3:
            hull = ConvexHull(np.array(pts))
            h_pts = np.vstack([np.array(pts)[hull.vertices], np.array(pts)[hull.vertices[0]]])
            poly = fols[t_name].newpolygon(name=f"BOUNDARY_{t_name}")
            poly.outerboundaryis = [(p[0], p[1]) for p in h_pts]
            poly.style.polystyle.color = simplekml.Color.changealphaint(50, simplekml.Color.gray)

    # Uncovered Folder
    fol_unc = kml.newfolder(name="❌ TIDAK_TERCOVER")
    for r in data['uncovered']:
        fol_unc.newpoint(name=f"UNC_{r.get('Name', 'H')}", coords=[(r.geometry.x, r.geometry.y)])

    df = pd.DataFrame(csv_list)
    st.divider()
    col_a, col_b = st.columns(2)
    with col_a: st.download_button("Download KMZ (With Boundary)", kml.kml().encode('utf-8'), "ISP_Frontage_Plan.kmz")
    with col_b: st.download_button("Download CSV (Full Data)", df.to_csv(index=False).encode('utf-8'), "Laporan_ISP.csv", "text/csv")
    st.dataframe(df, use_container_width=True)