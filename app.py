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
geolocator = ArcGIS(user_agent="isp_name_parser_pro")

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
    """
    Memisahkan Nama Tiang dan Kapasitas dari format: FBD057S01A01(20)
    Return: (kode_tiang, kapasitas)
    """
    # Mencari angka di dalam kurung
    match = re.search(r'^(.*?)\((\d+)\)$', str(full_name))
    if match:
        kode = match.group(1).strip()
        cap = int(match.group(2))
        return kode, cap
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
st.set_page_config(page_title="ISP Planner - Name Parser", layout="wide")
st.title("🌐 ISP Network Planner: Automated Slot & Name Parsing")

if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = None

st.sidebar.header("⚙️ Konfigurasi Standar")
max_dist = st.sidebar.slider("Radius Maksimal Rute (Meter)", 50, 500, 250)
default_cap = st.sidebar.number_input("Kapasitas Default (jika format salah)", 1, 32, 8)

c1, c2 = st.columns(2)
with c1: file_tiang = st.file_uploader("Upload KMZ Tiang", type=['kml', 'kmz'])
with c2: file_rumah = st.file_uploader("Upload KMZ Rumah", type=['kml', 'kmz'])

if file_tiang and file_rumah:
    gdf_tiang = load_data(file_tiang)
    gdf_rumah = load_data(file_rumah)

    if gdf_tiang is not None and gdf_rumah is not None:
        st.success(f"✅ Data Terdeteksi: {len(gdf_tiang)} Tiang & {len(gdf_rumah)} Rumah")
        
        if st.button("🚀 PROSES ANALISIS & PARSING NAMA"):
            with st.spinner("Mengekstrak kapasitas slot dan menghitung rute..."):
                avg_lat, avg_lon = gdf_tiang.geometry.y.mean(), gdf_tiang.geometry.x.mean()
                G = ox.graph_from_point((avg_lat, avg_lon), dist=3000, network_type='all')
                
                tiang_pts = np.array([[t.x, t.y] for t in gdf_tiang.geometry])
                rumah_pts = np.array([[r.x, r.y] for r in gdf_rumah.geometry])
                dist_mat = distance_matrix(rumah_pts, tiang_pts)
                
                # 1. Setup Tiang dengan Kapasitas Dinamis
                tiang_list = []
                for i, row in gdf_tiang.iterrows():
                    full_name = row.get('Name', f"T-{i}")
                    kode, cap = parse_pole_info(full_name, default_cap)
                    node = ox.distance.nearest_nodes(G, row.geometry.x, row.geometry.y)
                    tiang_list.append({
                        'id': i, 'kode': kode, 'cap': cap, 'load': 0, 
                        'node': node, 'coords': (row.geometry.x, row.geometry.y)
                    })

                # 2. Alokasi Global
                allocs = []
                taken = set()
                for _ in range(len(rumah_pts)):
                    if np.isnan(np.nanmin(dist_mat)): break
                    r_idx, t_idx = np.unravel_index(np.nanargmin(dist_mat), dist_mat.shape)
                    
                    if r_idx not in taken:
                        current_t = tiang_list[t_idx]
                        if current_t['load'] < current_t['cap']:
                            r_node = ox.distance.nearest_nodes(G, rumah_pts[r_idx][0], rumah_pts[r_idx][1])
                            try:
                                d = nx.shortest_path_length(G, current_t['node'], r_node, weight='length')
                                if d <= max_dist:
                                    path = nx.shortest_path(G, current_t['node'], r_node, weight='length')
                                    p_coords = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in path]
                                    p_coords.append((rumah_pts[r_idx][0], rumah_pts[r_idx][1]))
                                    
                                    allocs.append({
                                        't_kode': current_t['kode'], 'r_name': gdf_rumah.iloc[r_idx].get('Name', f"R-{r_idx}"),
                                        'lat': rumah_pts[r_idx][1], 'lon': rumah_pts[r_idx][0],
                                        'path': p_coords, 'dist': d, 't_idx': t_idx, 'cap': current_t['cap'],
                                        't_coords': current_t['coords']
                                    })
                                    taken.add(r_idx)
                                    tiang_list[t_idx]['load'] += 1
                            except: pass
                    dist_mat[r_idx, :] = np.nan

                # 3. Analisis Gagal
                uncovered = []
                for j, r in gdf_rumah.iterrows():
                    if j not in taken:
                        # Cek alasan
                        d_raw = distance_matrix([rumah_pts[j]], tiang_pts)[0]
                        c_idx = np.argmin(d_raw)
                        reason = "Radius > 250m"
                        if tiang_list[c_idx]['load'] >= tiang_list[c_idx]['cap']:
                            reason = f"Slot Full (Max: {tiang_list[c_idx]['cap']})"
                        uncovered.append({'name': r.get('Name', j), 'lat': r.geometry.y, 'lon': r.geometry.x, 'reason': reason})

                streets = batch_geocoding([(a['lat'], a['lon']) for a in allocs])
                st.session_state.analysis_data = {'alloc': allocs, 'streets': streets, 'unc': uncovered, 't_list': tiang_list}

# --- RENDER RESULTS ---
if st.session_state.analysis_data:
    res = st.session_state.analysis_data
    kml = simplekml.Kml()
    csv_rows = []
    fols = {}
    bound_pts = {}

    for idx, a in enumerate(res['alloc']):
        s_name = res['streets'][idx]
        csv_rows.append({
            'NO_RUMAH': a['r_name'], 'KODE_TIANG': a['t_kode'], 'MAX_SLOT': a['cap'],
            'ALAMAT_JALAN': s_name, 'JARAK_M': round(a['dist'], 1), 
            'LATITUDE': a['lat'], 'LONGITUDE': a['lon'], 'STATUS': 'TERCOVER'
        })
        
        if a['t_kode'] not in fols:
            fols[a['t_kode']] = kml.newfolder(name=f"AREA_{a['t_kode']}")
            fols[a['t_kode']].newpoint(name=f"CENTRAL_{a['t_kode']}({a['cap']})", coords=[a['t_coords']])
            bound_pts[a['t_kode']] = [a['t_coords']]
            
        bound_pts[a['t_kode']].append((a['lon'], a['lat']))
        p = fols[a['t_kode']].newpoint(name=a['r_name'], coords=[(a['lon'], a['lat'])])
        p.description = f"Tiang: {a['t_kode']}\nSlot: {a['cap']}\nAlamat: {s_name}"
        ls = fols[a['t_kode']].newlinestring(name=f"Line {a['r_name']}", coords=a['path'])
        ls.style.linestyle.width = 3

    # Poligon Wilayah
    for t_kode, pts in bound_pts.items():
        if len(pts) >= 3:
            hull = ConvexHull(np.array(pts))
            h_pts = np.vstack([np.array(pts)[hull.vertices], np.array(pts)[hull.vertices[0]]])
            poly = fols[t_kode].newpolygon(name=f"BOUND_{t_kode}")
            poly.outerboundaryis = [(p[0], p[1]) for p in h_pts]
            poly.style.polystyle.color = simplekml.Color.changealphaint(40, simplekml.Color.gray)

    # Uncovered
    fol_unc = kml.newfolder(name="❌ TIDAK_TERCOVER")
    for u in res['unc']:
        csv_rows.append({'NO_RUMAH': u['name'], 'KODE_TIANG': '-', 'MAX_SLOT': 0, 'ALAMAT_JALAN': 'N/A', 'JARAK_M': 0, 'LATITUDE': u['lat'], 'LONGITUDE': u['lon'], 'STATUS': 'GAGAL', 'ALASAN': u['reason']})
        up = fol_unc.newpoint(name=f"UNC_{u['name']}", coords=[(u['lon'], u['lat'])])
        up.description = f"Alasan: {u['reason']}"

    st.divider()
    st.subheader("📊 Statistik Hasil")
    c_m1, c_m2 = st.columns(2)
    c_m1.metric("Rumah Tercover", len(res['alloc']))
    c_m2.metric("Rumah Gagal", len(res['unc']))

    st.download_button("📥 Download KMZ (Map)", kml.kml().encode('utf-8'), "ISP_Project_Final.kmz")
    st.download_button("📥 Download CSV (Laporan)", pd.DataFrame(csv_rows).to_csv(index=False).encode('utf-8'), "Laporan_ISP.csv", "text/csv")
    st.dataframe(pd.DataFrame(csv_rows), use_container_width=True)