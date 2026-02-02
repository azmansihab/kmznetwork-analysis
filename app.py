import streamlit as st
import geopandas as gpd
import pandas as pd
import osmnx as ox
import networkx as nx
import simplekml
import zipfile
import numpy as np
import os
import re
from scipy.spatial import ConvexHull, distance_matrix
from geopy.geocoders import ArcGIS
from concurrent.futures import ThreadPoolExecutor

# Pengaturan OSMNX v2.0+
ox.settings.use_cache = True
ox.settings.log_console = False
geolocator = ArcGIS(user_agent="isp_balanced_optimizer")

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

def parse_pole_info(full_name, default_cap):
    # Regex untuk format: NamaTiang(Slot)
    match = re.search(r'^(.*?)\((\d+)\)$', str(full_name))
    if match:
        return match.group(1).strip(), int(match.group(2))
    return str(full_name), default_cap

@st.cache_data(show_spinner=False)
def get_street_name(lat, lon):
    try:
        location = geolocator.reverse(f"{lat}, {lon}", timeout=3)
        return location.raw['Address'].split(',')[0] if location else "Jalan Lokal"
    except: return "Jalan Lokal"

def batch_geocoding(coords_list):
    with ThreadPoolExecutor(max_workers=10) as executor:
        return list(executor.map(lambda p: get_street_name(p[0], p[1]), coords_list))

# --- UI APP ---
st.set_page_config(page_title="ISP Planner - Balanced Optimization", layout="wide")
st.title("🌐 ISP Network Planner: Balanced Global Optimization")

if 'persistent_data' not in st.session_state:
    st.session_state.persistent_data = None

st.sidebar.header("⚙️ Konfigurasi")
max_dist = st.sidebar.slider("Radius Maksimal Rute (Meter)", 50, 500, 250)
default_cap = st.sidebar.number_input("Kapasitas Default Slot", 1, 32, 16)

c1, c2 = st.columns(2)
with c1: file_tiang = st.file_uploader("Upload KMZ Tiang", type=['kml', 'kmz'])
with c2: file_rumah = st.file_uploader("Upload KMZ Rumah", type=['kml', 'kmz'])

if file_tiang and file_rumah:
    gdf_tiang = load_data(file_tiang)
    gdf_rumah = load_data(file_rumah)

    if gdf_tiang is not None and gdf_rumah is not None:
        st.success(f"✅ Data Terdeteksi: {len(gdf_tiang)} Tiang & {len(gdf_rumah)} Rumah")
        
        if st.button("🚀 JALANKAN OPTIMASI SEIMBANG"):
            with st.spinner("Mengoptimalkan distribusi rumah ke seluruh tiang..."):
                avg_lat, avg_lon = gdf_tiang.geometry.y.mean(), gdf_tiang.geometry.x.mean()
                G = ox.graph_from_point((avg_lat, avg_lon), dist=3000, network_type='all')
                
                tiang_pts = np.array([[t.x, t.y] for t in gdf_tiang.geometry])
                rumah_pts = np.array([[r.x, r.y] for r in gdf_rumah.geometry])
                
                # 1. Setup Tiang List
                tiang_list = []
                for i, row in gdf_tiang.iterrows():
                    kode, cap = parse_pole_info(row.get('Name', f"T-{i}"), default_cap)
                    node = ox.distance.nearest_nodes(G, row.geometry.x, row.geometry.y)
                    tiang_list.append({'id': i, 'kode': kode, 'cap': cap, 'load': 0, 'node': node, 'coords': (row.geometry.x, row.geometry.y)})

                # 2. Pre-calculate Semua Kemungkinan Rute (Semua Rumah ke Semua Tiang)
                all_possible_conns = []
                for j, r in gdf_rumah.iterrows():
                    r_node = ox.distance.nearest_nodes(G, r.geometry.x, r.geometry.y)
                    for t in tiang_list:
                        try:
                            d = nx.shortest_path_length(G, t['node'], r_node, weight='length')
                            if d <= max_dist:
                                all_possible_conns.append({
                                    'tiang_idx': t['id'], 'rumah_idx': j, 'dist': d,
                                    'r_coords': (r.geometry.x, r.geometry.y), 'r_node': r_node
                                })
                        except: continue

                # 3. Urutkan Secara Global untuk Distribusi yang Adil
                all_possible_conns = sorted(all_possible_conns, key=lambda x: x['dist'])
                
                final_allocs = []
                taken_homes = set()
                
                for conn in all_possible_conns:
                    t_idx = conn['tiang_idx']
                    r_idx = conn['rumah_idx']
                    current_t = tiang_list[t_idx]
                    
                    if r_idx not in taken_homes and current_t['load'] < current_t['cap']:
                        # Hitung Jalur Snap-to-Road
                        path = nx.shortest_path(G, current_t['node'], conn['r_node'], weight='length')
                        p_coords = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in path]
                        p_coords.append(conn['r_coords'])
                        
                        final_allocs.append({
                            't_name': current_t['kode'], 'r_name': gdf_rumah.iloc[r_idx].get('Name', f"R-{r_idx}"),
                            'lat': conn['r_coords'][1], 'lon': conn['r_coords'][0],
                            'path': p_coords, 'dist': conn['dist'], 't_idx': t_idx, 
                            'cap': current_t['cap'], 't_coord': current_t['coords']
                        })
                        taken_homes.add(r_idx)
                        tiang_list[t_idx]['load'] += 1

                # 4. Analisis Gagal
                uncovered = []
                for j, r in gdf_rumah.iterrows():
                    if j not in taken_homes:
                        uncovered.append({'name': r.get('Name', f"R-{j}"), 'lat': r.geometry.y, 'lon': r.geometry.x})

                streets = batch_geocoding([(a['lat'], a['lon']) for a in final_allocs])
                st.session_state.persistent_data = {'alloc': final_allocs, 'streets': streets, 'unc': uncovered}

# --- RENDER ---
if st.session_state.persistent_data:
    res = st.session_state.persistent_data
    kml = simplekml.Kml()
    csv_rows = []
    folders = {}
    hull_pts = {}

    for idx, a in enumerate(res['alloc']):
        s_name = res['streets'][idx]
        csv_rows.append({
            'NO_RUMAH': a['r_name'], 'TIANG': a['t_name'], 'SLOT': a['cap'],
            'ALAMAT': s_name, 'JARAK_M': round(a['dist'], 1), 
            'LATITUDE': a['lat'], 'LONGITUDE': a['lon']
        })
        
        if a['t_name'] not in folders:
            folders[a['t_name']] = kml.newfolder(name=f"AREA_{a['t_name']}")
            folders[a['t_name']].newpoint(name=f"CENTRAL_{a['t_name']}", coords=[a['t_coord']])
            hull_pts[a['t_name']] = [a['t_coord']]
            
        hull_pts[a['t_name']].append((a['lon'], a['lat']))
        p = folders[a['t_name']].newpoint(name=a['r_name'], coords=[(a['lon'], a['lat'])])
        ls = folders[a['t_name']].newlinestring(name=f"Cable {a['r_name']}", coords=a['path'])
        ls.style.linestyle.width = 3

    # Generate Boundary Polygons
    for t_name, pts in hull_pts.items():
        if len(pts) >= 3:
            hull = ConvexHull(np.array(pts))
            h_pts = np.vstack([np.array(pts)[hull.vertices], np.array(pts)[hull.vertices[0]]])
            poly = folders[t_name].newpolygon(name=f"BOUND_{t_name}")
            poly.outerboundaryis = [(p[0], p[1]) for p in h_pts]
            poly.style.polystyle.color = simplekml.Color.changealphaint(30, simplekml.Color.gray)

    st.divider()
    st.download_button("📥 Download KMZ (Balanced)", kml.kml().encode('utf-8'), "ISP_Balanced_Optimization.kmz")
    st.download_button("📥 Download CSV (Laporan)", pd.DataFrame(csv_rows).to_csv(index=False).encode('utf-8'), "Laporan_ISP.csv", "text/csv")
    st.dataframe(pd.DataFrame(csv_rows), use_container_width=True)