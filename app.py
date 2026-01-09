import streamlit as st
import geopandas as gpd
import pandas as pd
import osmnx as ox
import networkx as nx
import simplekml
import zipfile
import numpy as np
import os
from scipy.spatial import ConvexHull
from shapely.geometry import Point

# Update OSMNX Settings (v2.0+)
ox.settings.use_cache = True
ox.settings.log_console = False

def extract_kml_from_kmz(kmz_file):
    try:
        with zipfile.ZipFile(kmz_file, 'r') as z:
            kml_filename = [f for f in z.namelist() if f.endswith('.kml')][0]
            with z.open(kml_filename) as f:
                return f.read()
    except Exception as e:
        st.error(f"Gagal mengekstrak KMZ: {e}")
        return None

def load_data(uploaded_file):
    try:
        if uploaded_file.name.lower().endswith('.kmz'):
            kml_data = extract_kml_from_kmz(uploaded_file)
            if kml_data:
                temp_path = f"temp_{uploaded_file.name}.kml"
                with open(temp_path, "wb") as f:
                    f.write(kml_data)
                gdf = gpd.read_file(temp_path, driver='KML')
                if os.path.exists(temp_path): os.remove(temp_path)
                return gdf
        else:
            return gpd.read_file(uploaded_file, driver='KML')
    except Exception as e:
        st.error(f"Gagal membaca file {uploaded_file.name}: {e}")
        return None

st.set_page_config(page_title="ISP Network Analysis Pro", layout="wide")
st.title("🌐 ISP Network Analysis: Global Area Optimization")
st.markdown("""
Fitur Unggulan:
- **Global Optimization:** Mencegah garis rute menyilang antar wilayah tiang.
- **Road Network Routing:** Menghitung jarak maksimal 250m berdasarkan jalur jalan.
- **Dynamic Naming:** Folder dan Boundary otomatis mengikuti nama kode tiang di KMZ.
- **Uncovered Detection:** Menandai rumah yang tidak terjangkau.
""")

# Sidebar untuk parameter teknis
st.sidebar.header("⚙️ Parameter Teknis")
max_dist = st.sidebar.slider("Radius Maksimal Jalan (Meter)", 50, 1000, 250)
max_homes = st.sidebar.slider("Kapasitas Rumah per Tiang", 1, 32, 10)

col1, col2 = st.columns(2)
with col1:
    file_tiang = st.file_uploader("Upload KMZ/KML Tiang", type=['kml', 'kmz'], key="u_tiang")
with col2:
    file_rumah = st.file_uploader("Upload KMZ/KML Rumah", type=['kml', 'kmz'], key="u_rumah")

if file_tiang and file_rumah:
    gdf_tiang = load_data(file_tiang)
    gdf_rumah = load_data(file_rumah)

    if gdf_tiang is not None and gdf_rumah is not None:
        st.success(f"✅ Data Terdeteksi: {len(gdf_tiang)} Tiang & {len(gdf_rumah)} Rumah")
        
        if st.button("🚀 JALANKAN OPTIMASI WILAYAH"):
            with st.spinner("Sedang memproses optimasi global..."):
                # Ambil area peta jalan
                avg_lat, avg_lon = gdf_tiang.geometry.y.mean(), gdf_tiang.geometry.x.mean()
                G = ox.graph_from_point((avg_lat, avg_lon), dist=3000, network_type='drive')
                
                kml_out = simplekml.Kml()
                colors = [simplekml.Color.red, simplekml.Color.blue, simplekml.Color.green, 
                          simplekml.Color.yellow, simplekml.Color.purple, simplekml.Color.orange]

                # 1. PRA-PERHITUNGAN SEMUA KEMUNGKINAN RUTE (GLOBAL LIST)
                all_connections = []
                tiang_list = []
                
                for i, tiang in gdf_tiang.iterrows():
                    t_name = tiang.get('Name', f"T-{i+1}")
                    t_coord = (tiang.geometry.x, tiang.geometry.y)
                    t_node = ox.distance.nearest_nodes(G, t_coord[0], t_coord[1])
                    tiang_list.append({'id': i, 'name': t_name, 'node': t_node, 'coords': t_coord})

                for j, rumah in gdf_rumah.iterrows():
                    r_name = rumah.get('Name', f"H-{j+1}")
                    r_coord = (rumah.geometry.x, rumah.geometry.y)
                    r_node = ox.distance.nearest_nodes(G, r_coord[0], r_coord[1])
                    
                    for t in tiang_list:
                        try:
                            # Hitung jarak dasar untuk sorting awal
                            dist = nx.shortest_path_length(G, t['node'], r_node, weight='length')
                            if dist <= max_dist:
                                all_connections.append({
                                    'tiang_idx': t['id'], 'rumah_idx': j, 'dist': dist,
                                    'r_coords': r_coord, 'r_name': r_name, 't_name': t['name'],
                                    'r_node': r_node
                                })
                        except: continue

                # 2. OPTIMASI: URUTKAN BERDASARKAN JARAK TERPENDEK SECARA GLOBAL
                # Ini mencegah tiang mengambil rumah yang secara absolut lebih dekat ke tiang lain
                all_connections = sorted(all_connections, key=lambda x: x['dist'])
                
                final_allocations = {t['id']: [] for t in tiang_list}
                taken_homes = set()
                pole_load = {t['id']: 0 for t in tiang_list}

                for conn in all_connections:
                    t_idx = conn['tiang_idx']
                    r_idx = conn['rumah_idx']
                    
                    if r_idx not in taken_homes and pole_load[t_idx] < max_homes:
                        # Buat rute jalur jalan yang fix
                        t_node = tiang_list[t_idx]['node']
                        r_node = conn['r_node']
                        path = nx.shortest_path(G, t_node, r_node, weight='length')
                        line_coords = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in path]
                        
                        conn['path'] = line_coords
                        final_allocations[t_idx].append(conn)
                        taken_homes.add(r_idx)
                        pole_load[t_idx] += 1

                # 3. GENERATE VISUALISASI KML
                for t in tiang_list:
                    t_idx = t['id']
                    fol = kml_out.newfolder(name=f"AREA_{t['name']}")
                    fol.newpoint(name=f"PUSAT_{t['name']}", coords=[t['coords']])
                    
                    points_for_boundary = [t['coords']]
                    total_cable = 0

                    for res in final_allocations[t_idx]:
                        points_for_boundary.append(res['r_coords'])
                        total_cable += res['dist']
                        
                        # Titik Rumah
                        pr = fol.newpoint(name=f"{t['name']}-{res['r_name']}", coords=[res['r_coords']])
                        pr.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/target.png'
                        pr.style.iconstyle.scale = 0.5
                        
                        # Garis Rute
                        ls = fol.newlinestring(name=f"Rute {res['r_name']} ({int(res['dist'])}m)")
                        ls.coords = res['path']
                        ls.style.linestyle.color = colors[t_idx % len(colors)]
                        ls.style.linestyle.width = 3

                    # Boundary Area per Tiang
                    if len(points_for_boundary) >= 3:
                        pts_arr = np.array(points_for_boundary)
                        hull = ConvexHull(pts_arr)
                        hull_pts = np.vstack([pts_arr[hull.vertices], pts_arr[hull.vertices[0]]])
                        
                        poly = fol.newpolygon(name=f"BOUNDARY_{t['name']}")
                        poly.outerboundaryis = [(p[0], p[1]) for p in hull_pts]
                        poly.style.polystyle.color = simplekml.Color.changealphaint(50, colors[t_idx % len(colors)])
                        poly.style.linestyle.color = colors[t_idx % len(colors)]
                        poly.style.linestyle.width = 2

                # 4. FOLDER KHUSUS RUMAH TAK TERCOVER
                uncovered_fol = kml_out.newfolder(name="❌ TIDAK_TERCOVER")
                uncovered_count = 0
                for j, rumah in gdf_rumah.iterrows():
                    if j not in taken_homes:
                        p_unc = uncovered_fol.newpoint(name=f"UNCOVERED_{rumah.get('Name', j)}", 
                                                       coords=[(rumah.geometry.x, rumah.geometry.y)])
                        p_unc.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/forbidden.png'
                        uncovered_count += 1

                # 5. FINALISASI & DOWNLOAD
                output_path = "Analisis_ISP_Global_Optimized.kml"
                kml_out.save(output_path)
                
                st.balloons()
                st.subheader("📊 Ringkasan Analisis")
                c1, c2, c3 = st.columns(3)
                c1.metric("Rumah Terkoneksi", len(taken_homes))
                c2.metric("Rumah Tak Tercover", uncovered_count)
                c3.metric("Tiang Terpakai", len(gdf_tiang))

                with open(output_path, "rb") as f:
                    st.download_button("📥 DOWNLOAD HASIL OPTIMASI (KMZ)", f, file_name="ISP_Final_Analysis.kmz")