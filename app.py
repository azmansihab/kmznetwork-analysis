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

# Update OSMNX Settings
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
                with open(temp_path, "wb") as f: f.write(kml_data)
                gdf = gpd.read_file(temp_path, driver='KML')
                if os.path.exists(temp_path): os.remove(temp_path)
                return gdf
        else:
            return gpd.read_file(uploaded_file, driver='KML')
    except Exception as e:
        st.error(f"Gagal membaca file {uploaded_file.name}: {e}")
        return None

st.set_page_config(page_title="ISP Network Planner Pro", layout="wide")
st.title("🌐 ISP Network Planner: Anti-Overlap & Uncovered Analysis")

max_dist = st.sidebar.slider("Radius Maksimal Jalan (Meter)", 50, 1000, 250)
max_homes = st.sidebar.slider("Maksimal Rumah per Tiang", 1, 32, 10)

col1, col2 = st.columns(2)
with col1:
    file_tiang = st.file_uploader("Upload KMZ Tiang", type=['kml', 'kmz'])
with col2:
    file_rumah = st.file_uploader("Upload KMZ Rumah", type=['kml', 'kmz'])

if file_tiang and file_rumah:
    gdf_tiang = load_data(file_tiang)
    gdf_rumah = load_data(file_rumah)

    if gdf_tiang is not None and gdf_rumah is not None:
        st.success(f"✅ Data Terbaca: {len(gdf_tiang)} Tiang & {len(gdf_rumah)} Rumah.")
        
        if st.button("🚀 JALANKAN ANALISIS LENGKAP"):
            with st.spinner("Menghitung rute jalan dan alokasi rumah..."):
                avg_lat, avg_lon = gdf_tiang.geometry.y.mean(), gdf_tiang.geometry.x.mean()
                G = ox.graph_from_point((avg_lat, avg_lon), dist=3000, network_type='drive')
                
                kml_out = simplekml.Kml()
                colors = [simplekml.Color.red, simplekml.Color.blue, simplekml.Color.green, 
                          simplekml.Color.yellow, simplekml.Color.purple, simplekml.Color.orange]

                taken_homes_indices = set()

                # 1. PROSES ALOKASI RUMAH KE TIANG
                for i, tiang in gdf_tiang.iterrows():
                    kode_tiang = tiang.get('Name', f"Pole-{i+1}")
                    tiang_coord = (tiang.geometry.y, tiang.geometry.x)
                    tiang_node = ox.distance.nearest_nodes(G, tiang_coord[1], tiang_coord[0])
                    
                    fol = kml_out.newfolder(name=f"AREA_{kode_tiang}")
                    fol.newpoint(name=f"PUSAT_{kode_tiang}", coords=[(tiang.geometry.x, tiang.geometry.y)])

                    potential_homes = []
                    for j, rumah in gdf_rumah.iterrows():
                        if j in taken_homes_indices: continue
                            
                        rumah_coord = (rumah.geometry.y, rumah.geometry.x)
                        rumah_node = ox.distance.nearest_nodes(G, rumah_coord[1], rumah_coord[0])
                        
                        try:
                            dist = nx.shortest_path_length(G, tiang_node, rumah_node, weight='length')
                            if dist <= max_dist:
                                path = nx.shortest_path(G, tiang_node, rumah_node, weight='length')
                                line_coords = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in path]
                                potential_homes.append({
                                    'index': j, 'dist': dist, 'coords': (rumah.geometry.x, rumah.geometry.y),
                                    'path': line_coords, 'name': rumah.get('Name', f"H-{j}")
                                })
                        except: continue

                    valid_homes = sorted(potential_homes, key=lambda x: x['dist'])[:max_homes]
                    points_for_boundary = [(tiang.geometry.x, tiang.geometry.y)]
                    
                    for home in valid_homes:
                        taken_homes_indices.add(home['index'])
                        points_for_boundary.append(home['coords'])
                        p_rmh = fol.newpoint(name=f"{kode_tiang}-{home['name']}", coords=[home['coords']])
                        p_rmh.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/paddle/wht-circle.png'
                        
                        ls = fol.newlinestring(name=f"Route {home['name']} ({int(home['dist'])}m)")
                        ls.coords = home['path']
                        ls.style.linestyle.color = colors[i % len(colors)]
                        ls.style.linestyle.width = 3

                    if len(points_for_boundary) >= 3:
                        pts_arr = np.array(points_for_boundary)
                        hull = ConvexHull(pts_arr)
                        hull_pts = np.vstack([pts_arr[hull.vertices], pts_arr[hull.vertices[0]]])
                        poly = fol.newpolygon(name=f"BOUNDARY_{kode_tiang}")
                        poly.outerboundaryis = [(p[0], p[1]) for p in hull_pts]
                        poly.style.polystyle.color = simplekml.Color.changealphaint(40, colors[i % len(colors)])

                # 2. PROSES RUMAH YANG TIDAK TERCOVER
                uncovered_fol = kml_out.newfolder(name="❌ TIDAK_TERCOVER")
                uncovered_count = 0
                for j, rumah in gdf_rumah.iterrows():
                    if j not in taken_homes_indices:
                        p_unc = uncovered_fol.newpoint(name=f"UNCOVERED_{rumah.get('Name', j)}", 
                                                       coords=[(rumah.geometry.x, rumah.geometry.y)])
                        # Ikon Silang Merah untuk yang tidak tercover
                        p_unc.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/paddle/red-square.png'
                        uncovered_count += 1

                # 3. OUTPUT & STATISTIK
                output_path = "Analisis_ISP_Lengkap.kml"
                kml_out.save(output_path)
                
                st.balloons()
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Rumah", len(gdf_rumah))
                c2.metric("Tercover", len(taken_homes_indices), delta_color="normal")
                c3.metric("Tidak Tercover", uncovered_count, delta="-"+str(uncovered_count), delta_color="inverse")

                with open(output_path, "rb") as f:
                    st.download_button("📥 DOWNLOAD HASIL KMZ (LENGKAP)", f, file_name="Analisis_ISP_Final.kmz")