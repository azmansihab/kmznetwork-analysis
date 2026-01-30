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

# Pengaturan OSMNX (v2.0+)
ox.settings.use_cache = True
ox.settings.log_console = False

# Inisialisasi Geocoding ArcGIS
geolocator = ArcGIS(user_agent="isp_planner_final_fixed_v2")

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
        st.error(f"Error membaca file: {e}")
        return None

@st.cache_data(show_spinner=False)
def get_street_name(lat, lon):
    try:
        location = geolocator.reverse(f"{lat}, {lon}", timeout=3)
        return location.raw['Address'].split(',')[0] if location else "Jalan Lokal"
    except: return "Jalan Lokal"

def batch_geocoding(coords_list):
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(lambda p: get_street_name(p[0], p[1]), coords_list))
    return results

st.set_page_config(page_title="ISP Planner - Persistent Data", layout="wide")
st.title("🌐 ISP Network Planner: Full Analysis & Boundary")

# Inisialisasi Session State agar data tidak hilang saat download
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

st.sidebar.header("⚙️ Konfigurasi")
max_dist = st.sidebar.slider("Radius Maksimal (Meter)", 50, 1000, 250)
max_homes = st.sidebar.slider("Kapasitas Rumah per Tiang", 1, 32, 10)

col1, col2 = st.columns(2)
with col1: file_tiang = st.file_uploader("Upload Data Tiang (KMZ/KML)", type=['kml', 'kmz'])
with col2: file_rumah = st.file_uploader("Upload Data Rumah (KMZ/KML)", type=['kml', 'kmz'])

if file_tiang and file_rumah:
    gdf_tiang = load_data(file_tiang)
    gdf_rumah = load_data(file_rumah)

    if gdf_tiang is not None and gdf_rumah is not None:
        st.success(f"✅ Data Siap: {len(gdf_tiang)} Tiang & {len(gdf_rumah)} Rumah.")
        
        if st.button("🚀 PROSES DATA SEKARANG"):
            with st.spinner("Menghitung rute, boundary, dan koordinat..."):
                avg_lat, avg_lon = gdf_tiang.geometry.y.mean(), gdf_tiang.geometry.x.mean()
                G = ox.graph_from_point((avg_lat, avg_lon), dist=3000, network_type='all')
                
                tiang_coords = np.array([[t.x, t.y] for t in gdf_tiang.geometry])
                rumah_coords = np.array([[r.x, r.y] for r in gdf_rumah.geometry])
                dist_mat = distance_matrix(rumah_coords, tiang_coords)
                
                final_allocations = []
                taken_homes = set()
                pole_capacities = {i: 0 for i in range(len(tiang_coords))}
                
                # Alokasi Global
                for _ in range(len(rumah_coords)):
                    min_val = np.nanmin(dist_mat)
                    if np.isnan(min_val): break
                    r_idx, t_idx = np.unravel_index(np.nanargmin(dist_mat), dist_mat.shape)
                    
                    if r_idx not in taken_homes and pole_capacities[t_idx] < max_homes:
                        r_node = ox.distance.nearest_nodes(G, rumah_coords[r_idx][0], rumah_coords[r_idx][1])
                        t_node = ox.distance.nearest_nodes(G, tiang_coords[t_idx][0], tiang_coords[t_idx][1])
                        try:
                            net_dist = nx.shortest_path_length(G, t_node, r_node, weight='length')
                            if net_dist <= max_dist:
                                path = nx.shortest_path(G, t_node, r_node, weight='length')
                                p_coords = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in path]
                                final_allocations.append({
                                    'tiang_name': gdf_tiang.iloc[t_idx].get('Name', f"T-{t_idx}"),
                                    'rumah_name': gdf_rumah.iloc[r_idx].get('Name', f"R-{r_idx}"),
                                    'lat': rumah_coords[r_idx][1], 'lon': rumah_coords[r_idx][0],
                                    'path': p_coords, 'dist': net_dist, 'tiang_idx': t_idx,
                                    'tiang_coord': (tiang_coords[t_idx][0], tiang_coords[t_idx][1])
                                })
                                taken_homes.add(r_idx)
                                pole_capacities[t_idx] += 1
                        except: pass
                    dist_mat[r_idx, :] = np.nan

                # Geocoding
                geo_req = [(a['lat'], a['lon']) for a in final_allocations]
                street_results = batch_geocoding(geo_req)
                
                # Simpan ke Session State
                st.session_state.analysis_result = {
                    'allocations': final_allocations,
                    'streets': street_results,
                    'uncovered': [gdf_rumah.iloc[j] for j in range(len(gdf_rumah)) if j not in taken_homes],
                    'tiang_coords': tiang_coords
                }

# Cek apakah hasil analisis tersedia di session state
if st.session_state.analysis_result:
    res = st.session_state.analysis_result
    allocs = res['allocations']
    streets = res['streets']
    
    kml_out = simplekml.Kml()
    colors = [simplekml.Color.red, simplekml.Color.blue, simplekml.Color.green, simplekml.Color.yellow, simplekml.Color.purple]
    csv_list = []
    folders = {}
    points_per_pole = {} # Untuk boundary

    for idx, alloc in enumerate(allocs):
        s_name = streets[idx]
        t_name = alloc['tiang_name']
        t_idx = alloc['tiang_idx']
        
        csv_list.append({
            'NO_RUMAH': alloc['rumah_name'], 'TIANG_TERHUBUNG': t_name, 'NAMA_JALAN': s_name,
            'JARAK_KABEL_M': round(alloc['dist'], 2), 'LATITUDE': alloc['lat'], 'LONGITUDE': alloc['lon']
        })
        
        if t_name not in folders:
            folders[t_name] = kml_out.newfolder(name=f"AREA_{t_name}")
            folders[t_name].newpoint(name=f"PUSAT_{t_name}", coords=[alloc['tiang_coord']])
            points_per_pole[t_name] = [alloc['tiang_coord']]
            
        points_per_pole[t_name].append((alloc['lon'], alloc['lat']))
        p = folders[t_name].newpoint(name=alloc['rumah_name'], coords=[(alloc['lon'], alloc['lat'])])
        p.description = f"Jalan: {s_name}\nJarak: {int(alloc['dist'])}m"
        ls = folders[t_name].newlinestring(name=f"Kabel {alloc['rumah_name']}", coords=alloc['path'])
        ls.style.linestyle.color = colors[t_idx % len(colors)]
        ls.style.linestyle.width = 3

    # Generate Boundary Polygons
    for t_name, pts in points_per_pole.items():
        if len(pts) >= 3:
            pts_arr = np.array(pts)
            hull = ConvexHull(pts_arr)
            hull_pts = np.vstack([pts_arr[hull.vertices], pts_arr[hull.vertices[0]]])
            poly = folders[t_name].newpolygon(name=f"BOUNDARY_{t_name}")
            poly.outerboundaryis = [(p[0], p[1]) for p in hull_pts]
            # Ambil indeks warna berdasarkan tiang pertama yang ditemukan di alokasi
            t_idx_for_color = next(a['tiang_idx'] for a in allocs if a['tiang_name'] == t_name)
            poly.style.polystyle.color = simplekml.Color.changealphaint(50, colors[t_idx_for_color % len(colors)])

    # Uncovered
    fol_unc = kml_out.newfolder(name="❌ TIDAK_TERCOVER")
    for r in res['uncovered']:
        fol_unc.newpoint(name=f"UNC_{r.get('Name', 'Home')}", coords=[(r.geometry.x, r.geometry.y)])

    df_final = pd.DataFrame(csv_list)
    
    st.divider()
    st.subheader("📥 Hasil Analisis (Tersimpan)")
    c1, c2 = st.columns(2)
    with c1: st.download_button("Download KMZ (Dengan Boundary)", kml_out.kml().encode('utf-8'), "ISP_Map_With_Boundary.kmz")
    with c2: st.download_button("Download CSV (Laporan)", df_final.to_csv(index=False).encode('utf-8'), "Laporan_ISP.csv", "text/csv")
    
    st.subheader("📊 Tabel Laporan")
    st.dataframe(df_final, use_container_width=True)