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
geolocator = ArcGIS(user_agent="isp_pro_optimizer_v2")

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
        st.error(f"Error Loading: {e}")
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
st.set_page_config(page_title="ISP Planner Pro Final", layout="wide")
st.title("🌐 ISP Network Planner: Analisis Frontage & Koordinat")

if 'final_results' not in st.session_state:
    st.session_state.final_results = None

st.sidebar.header("⚙️ Konfigurasi")
max_dist = st.sidebar.slider("Radius Maksimal (Meter)", 50, 500, 250)
max_homes = st.sidebar.slider("Kapasitas per Tiang", 1, 32, 16)

c1, c2 = st.columns(2)
with c1: file_tiang = st.file_uploader("Upload Data Tiang", type=['kml', 'kmz'])
with c2: file_rumah = st.file_uploader("Upload Data Rumah", type=['kml', 'kmz'])

if file_tiang and file_rumah:
    gdf_tiang = load_data(file_tiang)
    gdf_rumah = load_data(file_rumah)

    if gdf_tiang is not None and gdf_rumah is not None:
        st.success(f"✅ Data Terbaca: {len(gdf_tiang)} Tiang & {len(gdf_rumah)} Rumah")
        
        if st.button("🚀 JALANKAN ANALISIS SEKARANG"):
            with st.spinner("Memproses optimasi jalur dan geocoding koordinat..."):
                avg_lat, avg_lon = gdf_tiang.geometry.y.mean(), gdf_tiang.geometry.x.mean()
                G = ox.graph_from_point((avg_lat, avg_lon), dist=3000, network_type='all')
                
                tiang_pts = np.array([[t.x, t.y] for t in gdf_tiang.geometry])
                rumah_pts = np.array([[r.x, r.y] for r in gdf_rumah.geometry])
                dist_mat = distance_matrix(rumah_pts, tiang_pts)
                
                allocs = []
                uncovered = []
                taken = set()
                load = {i: 0 for i in range(len(tiang_pts))}
                
                # Optimasi Global (Anti-Crossing)
                for _ in range(len(rumah_pts)):
                    if np.isnan(np.nanmin(dist_mat)): break
                    r_idx, t_idx = np.unravel_index(np.nanargmin(dist_mat), dist_mat.shape)
                    
                    if r_idx not in taken:
                        if load[t_idx] < max_homes:
                            r_node = ox.distance.nearest_nodes(G, rumah_pts[r_idx][0], rumah_pts[r_idx][1])
                            t_node = ox.distance.nearest_nodes(G, tiang_pts[t_idx][0], tiang_pts[t_idx][1])
                            try:
                                d = nx.shortest_path_length(G, t_node, r_node, weight='length')
                                if d <= max_dist:
                                    path = nx.shortest_path(G, t_node, r_node, weight='length')
                                    p_coords = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in path]
                                    p_coords.append((rumah_pts[r_idx][0], rumah_pts[r_idx][1]))
                                    
                                    allocs.append({
                                        't_name': gdf_tiang.iloc[t_idx].get('Name', f"T-{t_idx}"),
                                        'r_name': gdf_rumah.iloc[r_idx].get('Name', f"R-{r_idx}"),
                                        'lat': rumah_pts[r_idx][1], 'lon': rumah_pts[r_idx][0],
                                        'path': p_coords, 'dist': d, 't_idx': t_idx,
                                        't_coord': (tiang_pts[t_idx][0], tiang_pts[t_idx][1])
                                    })
                                    taken.add(r_idx)
                                    load[t_idx] += 1
                            except: pass
                    dist_mat[r_idx, :] = np.nan

                # Deteksi Alasan Tidak Tercover
                for j, r in gdf_rumah.iterrows():
                    if j not in taken:
                        reason = "Melebihi Radius 250m"
                        # Cari tiang terdekat berdasarkan koordinat mentah
                        dist_to_poles = distance_matrix([rumah_pts[j]], tiang_pts)[0]
                        closest_pole = np.argmin(dist_to_poles)
                        if load[closest_pole] >= max_homes:
                            reason = "Kapasitas Tiang Penuh"
                        
                        uncovered.append({
                            'r_name': r.get('Name', f"R-{j}"),
                            'lat': r.geometry.y, 'lon': r.geometry.x,
                            'reason': reason
                        })

                # Batch Geocoding
                streets = batch_geocoding([(a['lat'], a['lon']) for a in allocs])
                
                st.session_state.final_results = {
                    'alloc': allocs, 'streets': streets, 'uncovered': uncovered,
                    'total_h': len(gdf_rumah), 'total_t': len(gdf_tiang)
                }

# --- Visualisasi & Laporan ---
if st.session_state.final_results:
    res = st.session_state.final_results
    kml = simplekml.Kml()
    palette = [simplekml.Color.cyan, simplekml.Color.magenta, simplekml.Color.lime, simplekml.Color.orange]
    csv_data = []
    folders = {}
    hull_pts = {}

    for idx, a in enumerate(res['alloc']):
        s_name = res['streets'][idx]
        csv_data.append({
            'NO_RUMAH': a['r_name'], 'TIANG': a['t_name'], 'ALAMAT': s_name,
            'PANJANG_KABEL_M': round(a['dist'], 1), 'LATITUDE': a['lat'], 'LONGITUDE': a['lon'],
            'STATUS': 'Tercover', 'ALASAN': '-'
        })
        
        if a['t_name'] not in folders:
            folders[a['t_name']] = kml.newfolder(name=f"AREA_{a['t_name']}")
            folders[a['t_name']].newpoint(name=f"CENTRAL_{a['t_name']}", coords=[a['t_coord']])
            hull_pts[a['t_name']] = [a['t_coord']]
            
        hull_pts[a['t_name']].append((a['lon'], a['lat']))
        p = folders[a['t_name']].newpoint(name=a['r_name'], coords=[(a['lon'], a['lat'])])
        p.description = f"Tiang: {a['t_name']}\nJalan: {s_name}\nKoordinat: {a['lat']}, {a['lon']}"
        
        ls = folders[a['t_name']].newlinestring(name=f"Route {a['r_name']}", coords=a['path'])
        ls.style.linestyle.color = palette[a['t_idx'] % len(palette)]
        ls.style.linestyle.width = 3

    # Boundary
    for t_name, pts in hull_pts.items():
        if len(pts) >= 3:
            hull = ConvexHull(np.array(pts))
            h_pts = np.vstack([np.array(pts)[hull.vertices], np.array(pts)[hull.vertices[0]]])
            poly = folders[t_name].newpolygon(name=f"COVERAGE_{t_name}")
            poly.outerboundaryis = [(p[0], p[1]) for p in h_pts]
            poly.style.polystyle.color = simplekml.Color.changealphaint(30, simplekml.Color.gray)

    # Folder Tidak Tercover
    fol_unc = kml.newfolder(name="❌ TIDAK_TERCOVER")
    for u in res['uncovered']:
        csv_data.append({
            'NO_RUMAH': u['r_name'], 'TIANG': '-', 'ALAMAT': 'N/A',
            'PANJANG_KABEL_M': 0, 'LATITUDE': u['lat'], 'LONGITUDE': u['lon'],
            'STATUS': 'Gagal', 'ALASAN': u['reason']
        })
        up = fol_unc.newpoint(name=f"UNC_{u['r_name']}", coords=[(u['lon'], u['lat'])])
        up.description = f"Alasan: {u['reason']}"

    df_final = pd.DataFrame(csv_data)
    st.divider()
    
    # Statistik Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Rumah", res['total_h'])
    m2.metric("Berhasil Tercover", len(res['alloc']))
    m3.metric("Gagal Tercover", len(res['uncovered']), delta_color="inverse")

    dl1, dl2 = st.columns(2)
    with dl1: st.download_button("📥 Download KMZ (Boundary + Jalur)", kml.kml().encode('utf-8'), "ISP_Analysis.kmz")
    with dl2: st.download_button("📥 Download CSV (Data & Koordinat)", df_final.to_csv(index=False).encode('utf-8'), "Laporan_ISP.csv", "text/csv")
    
    st.subheader("📊 Tabel Analisis Terperinci")
    st.dataframe(df_final, use_container_width=True)