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

def get_street_names(G, path):
    """Mengambil nama jalan dengan pengecekan atribut yang lebih luas."""
    names = []
    # Loop melalui setiap edge dalam path
    edges = ox.utils_graph.get_route_edge_attributes(G, path)
    
    for edge in edges:
        if 'name' in edge:
            name = edge['name']
            if isinstance(name, list):
                names.extend([str(n) for n in name])
            else:
                names.append(str(name))
        elif 'ref' in edge: # Jika nama tidak ada, coba ambil referensi jalan (misal: kode jalan)
            names.append(str(edge['ref']))
            
    # Membersihkan list dan mengambil nama unik
    unique_names = sorted(list(set([n for n in names if n])))
    return ", ".join(unique_names) if unique_names else "Jalan lokal/tidak bernama"

st.set_page_config(page_title="ISP Network Planner Pro", layout="wide")
st.title("🌐 ISP Network Planner: Full Spatial & Street Analysis")

# Sidebar
st.sidebar.header("⚙️ Parameter")
max_dist = st.sidebar.slider("Radius Maksimal Jalan (Meter)", 50, 1000, 250)
max_homes = st.sidebar.slider("Kapasitas Rumah per Tiang", 1, 32, 10)

col1, col2 = st.columns(2)
with col1:
    file_tiang = st.file_uploader("Upload KMZ/KML Tiang", type=['kml', 'kmz'])
with col2:
    file_rumah = st.file_uploader("Upload KMZ/KML Rumah", type=['kml', 'kmz'])

if file_tiang and file_rumah:
    gdf_tiang = load_data(file_tiang)
    gdf_rumah = load_data(file_rumah)

    if gdf_tiang is not None and gdf_rumah is not None:
        st.success(f"✅ Data Terdeteksi: {len(gdf_tiang)} Tiang & {len(gdf_rumah)} Rumah")
        
        if st.button("🚀 JALANKAN ANALISIS OPTIMASI LENGKAP"):
            with st.spinner("Mengunduh data jalan (termasuk jalan setapak) dan memproses rute..."):
                avg_lat, avg_lon = gdf_tiang.geometry.y.mean(), gdf_tiang.geometry.x.mean()
                
                # network_type='all' akan mengambil semua jalur (mobil, motor, jalan kaki)
                # agar jalan kecil di pemukiman masuk ke sistem
                G = ox.graph_from_point((avg_lat, avg_lon), dist=3000, network_type='all')
                
                kml_out = simplekml.Kml()
                colors = [simplekml.Color.red, simplekml.Color.blue, simplekml.Color.green, 
                          simplekml.Color.yellow, simplekml.Color.purple, simplekml.Color.orange]

                all_connections = []
                tiang_list = []
                csv_data = [] 
                
                for i, tiang in gdf_tiang.iterrows():
                    t_name = tiang.get('Name', f"T-{i+1}")
                    t_node = ox.distance.nearest_nodes(G, tiang.geometry.x, tiang.geometry.y)
                    tiang_list.append({'id': i, 'name': t_name, 'node': t_node, 'coords': (tiang.geometry.x, tiang.geometry.y)})

                for j, rumah in gdf_rumah.iterrows():
                    r_node = ox.distance.nearest_nodes(G, rumah.geometry.x, rumah.geometry.y)
                    for t in tiang_list:
                        try:
                            dist = nx.shortest_path_length(G, t['node'], r_node, weight='length')
                            if dist <= max_dist:
                                all_connections.append({
                                    'tiang_idx': t['id'], 'rumah_idx': j, 'dist': dist,
                                    'r_coords': (rumah.geometry.x, rumah.geometry.y), 
                                    'r_name': rumah.get('Name', f"H-{j+1}"),
                                    'r_node': r_node
                                })
                        except: continue

                all_connections = sorted(all_connections, key=lambda x: x['dist'])
                taken_homes = set()
                pole_load = {t['id']: 0 for t in tiang_list}
                final_allocations = {t['id']: [] for t in tiang_list}

                for conn in all_connections:
                    t_idx = conn['tiang_idx']
                    r_idx = conn['rumah_idx']
                    if r_idx not in taken_homes and pole_load[t_idx] < max_homes:
                        t_node = tiang_list[t_idx]['node']
                        path = nx.shortest_path(G, t_node, conn['r_node'], weight='length')
                        
                        street_names = get_street_names(G, path)
                        
                        conn['path'] = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in path]
                        conn['streets'] = street_names
                        
                        final_allocations[t_idx].append(conn)
                        taken_homes.add(r_idx)
                        pole_load[t_idx] += 1
                        
                        csv_data.append({
                            'KODE_TIANG': tiang_list[t_idx]['name'],
                            'NAMA_RUMAH': conn['r_name'],
                            'NAMA_JALAN': street_names,
                            'JARAK_KABEL_M': round(conn['dist'], 2),
                            'LONGITUDE': conn['r_coords'][0],
                            'LATITUDE': conn['r_coords'][1]
                        })

                # Bangun KML
                for t in tiang_list:
                    t_idx = t['id']
                    fol = kml_out.newfolder(name=f"AREA_{t['name']}")
                    fol.newpoint(name=f"PUSAT_{t['name']}", coords=[t['coords']])
                    points_for_boundary = [t['coords']]

                    for res in final_allocations[t_idx]:
                        points_for_boundary.append(res['r_coords'])
                        p_rmh = fol.newpoint(name=f"{res['r_name']}", coords=[res['r_coords']])
                        p_rmh.description = f"Melewati: {res['streets']}\nJarak: {int(res['dist'])}m"
                        
                        ls = fol.newlinestring(name=f"Rute ke {res['r_name']}")
                        ls.coords = res['path']
                        ls.style.linestyle.color = colors[t_idx % len(colors)]
                        ls.style.linestyle.width = 3

                    if len(points_for_boundary) >= 3:
                        pts_arr = np.array(points_for_boundary)
                        hull = ConvexHull(pts_arr)
                        hull_pts = np.vstack([pts_arr[hull.vertices], pts_arr[hull.vertices[0]]])
                        poly = fol.newpolygon(name=f"BOUNDARY_{t['name']}")
                        poly.outerboundaryis = [(p[0], p[1]) for p in hull_pts]
                        poly.style.polystyle.color = simplekml.Color.changealphaint(50, colors[t_idx % len(colors)])

                for j, rumah in gdf_rumah.iterrows():
                    if j not in taken_homes:
                        csv_data.append({
                            'KODE_TIANG': 'TIDAK TERCOVER',
                            'NAMA_RUMAH': rumah.get('Name', f"H-{j+1}"),
                            'NAMA_JALAN': 'N/A',
                            'JARAK_KABEL_M': 0,
                            'LONGITUDE': rumah.geometry.x,
                            'LATITUDE': rumah.geometry.y
                        })

                df_csv = pd.DataFrame(csv_data)
                kml_path = "ISP_Analysis_Final.kml"
                kml_out.save(kml_path)

                st.balloons()
                st.subheader("📥 Download Hasil Analisis")
                c1, c2 = st.columns(2)
                with c1:
                    with open(kml_path, "rb") as f:
                        st.download_button("Download KMZ (Peta + Nama Jalan)", f, file_name="ISP_Analysis.kmz")
                with c2:
                    csv_file = df_csv.to_csv(index=False).encode('utf-8')
                    st.download_button("Download CSV (Data Teknis)", csv_file, "Data_ISP.csv", "text/csv")
                
                st.dataframe(df_csv)