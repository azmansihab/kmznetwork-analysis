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

# Update OSMNX Settings
ox.settings.use_cache = True
ox.settings.log_console = False

def extract_kml_from_kmz(kmz_file):
    try:
        with zipfile.ZipFile(kmz_file, 'r') as z:
            kml_filename = [f for f in z.namelist() if f.endswith('.kml')][0]
            with z.open(kml_filename) as f: return f.read()
    except: return None

def load_data(uploaded_file):
    if uploaded_file.name.lower().endswith('.kmz'):
        kml_data = extract_kml_from_kmz(uploaded_file)
        if kml_data:
            temp_path = f"temp_{uploaded_file.name}.kml"
            with open(temp_path, "wb") as f: f.write(kml_data)
            gdf = gpd.read_file(temp_path, driver='KML')
            if os.path.exists(temp_path): os.remove(temp_path)
            return gdf
    return gpd.read_file(uploaded_file, driver='KML')

st.set_page_config(page_title="ISP Planner - Neat Allocation", layout="wide")
st.title("🌐 ISP Network Planner: Neat & Organized Allocation")
st.markdown("Fokus: **Mencegah Garis Bersilangan** dan **Optimasi Wilayah Cluster**.")

max_dist = st.sidebar.slider("Radius Maksimal (Meter)", 50, 1000, 250)
max_homes = st.sidebar.slider("Kapasitas Rumah per Tiang", 1, 32, 10)

col1, col2 = st.columns(2)
with col1: file_tiang = st.file_uploader("Upload Tiang (Pole)", type=['kml', 'kmz'])
with col2: file_rumah = st.file_uploader("Upload Rumah (Homepass)", type=['kml', 'kmz'])

if file_tiang and file_rumah:
    gdf_tiang = load_data(file_tiang)
    gdf_rumah = load_data(file_rumah)

    if gdf_tiang is not None and gdf_rumah is not None:
        st.success(f"✅ Data Terdeteksi: {len(gdf_tiang)} Tiang & {len(gdf_rumah)} Rumah.")
        
        if st.button("🚀 MULAI ANALISIS (VERSI RAPI)"):
            with st.spinner("Mengatur wilayah agar tidak loncat-loncat..."):
                # 1. Download Graph Jalan
                avg_lat, avg_lon = gdf_tiang.geometry.y.mean(), gdf_tiang.geometry.x.mean()
                G = ox.graph_from_point((avg_lat, avg_lon), dist=2500, network_type='all')
                
                # 2. Pre-processing Data Spasial
                tiang_coords = np.array([[t.x, t.y] for t in gdf_tiang.geometry])
                rumah_coords = np.array([[r.x, r.y] for r in gdf_rumah.geometry])
                
                # Hitung Matrix Jarak Euclidean sebagai referensi awal wilayah
                dist_mat = distance_matrix(rumah_coords, tiang_coords)
                
                # 3. Alokasi dengan Logika "Wilayah Kekuasaan"
                # Kita urutkan rumah dari yang paling 'jelas' miliknya tiang tertentu
                final_allocations = {i: [] for i in range(len(tiang_coords))}
                taken_homes = set()
                pole_capacities = {i: 0 for i in range(len(tiang_coords))}
                
                # Loop untuk mengalokasikan rumah ke tiang terdekat secara global
                # Strategi: Tiang hanya mengambil rumah yang memang paling dekat dengannya dibanding tiang lain
                for _ in range(len(rumah_coords)):
                    min_val = np.nanmin(dist_mat)
                    if np.isnan(min_val): break
                    
                    r_idx, t_idx = np.unravel_index(np.nanargmin(dist_mat), dist_mat.shape)
                    
                    # Jika rumah belum diambil dan kapasitas tiang masih ada
                    if r_idx not in taken_homes and pole_capacities[t_idx] < max_homes:
                        # Cek jarak jalan asli (Network Distance)
                        r_node = ox.distance.nearest_nodes(G, rumah_coords[r_idx][0], rumah_coords[r_idx][1])
                        t_node = ox.distance.nearest_nodes(G, tiang_coords[t_idx][0], tiang_coords[t_idx][1])
                        
                        try:
                            net_dist = nx.shortest_path_length(G, t_node, r_node, weight='length')
                            if net_dist <= max_dist:
                                path = nx.shortest_path(G, t_node, r_node, weight='length')
                                l_coords = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in path]
                                
                                final_allocations[t_idx].append({
                                    'r_name': gdf_rumah.iloc[r_idx].get('Name', f"H-{r_idx}"),
                                    'coords': rumah_coords[r_idx],
                                    'path': l_coords,
                                    'dist': net_dist
                                })
                                taken_homes.add(r_idx)
                                pole_capacities[t_idx] += 1
                        except: pass
                    
                    # Matikan pilihan ini agar tidak diproses lagi
                    dist_mat[r_idx, :] = np.nan

                # 4. Generate KMZ
                kml_out = simplekml.Kml()
                colors = [simplekml.Color.red, simplekml.Color.blue, simplekml.Color.green, simplekml.Color.yellow, simplekml.Color.purple, simplekml.Color.orange]
                
                for i, t in gdf_tiang.iterrows():
                    kode_tiang = t.get('Name', f"T-{i}")
                    fol = kml_out.newfolder(name=f"AREA_{kode_tiang}")
                    fol.newpoint(name=f"PUSAT_{kode_tiang}", coords=[(t.geometry.x, t.geometry.y)])
                    
                    pts_boundary = [(t.geometry.x, t.geometry.y)]
                    for res in final_allocations[i]:
                        pts_boundary.append((res['coords'][0], res['coords'][1]))
                        p = fol.newpoint(name=res['r_name'], coords=[(res['coords'][0], res['coords'][1])])
                        ls = fol.newlinestring(name=f"Route {res['r_name']}", coords=res['path'])
                        ls.style.linestyle.color = colors[i % len(colors)]
                        ls.style.linestyle.width = 3
                    
                    if len(pts_boundary) >= 3:
                        hull = ConvexHull(np.array(pts_boundary))
                        hull_pts = np.vstack([np.array(pts_boundary)[hull.vertices], np.array(pts_boundary)[hull.vertices[0]]])
                        poly = fol.newpolygon(name=f"BOUNDARY_{kode_tiang}")
                        poly.outerboundaryis = [(p[0], p[1]) for p in hull_pts]
                        poly.style.polystyle.color = simplekml.Color.changealphaint(50, colors[i % len(colors)])

                st.balloons()
                st.download_button("📥 Download Hasil (Versi Rapi)", kml_out.kml().encode('utf-8'), "ISP_Neat_Plan.kmz")