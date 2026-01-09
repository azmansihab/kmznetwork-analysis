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
    with zipfile.ZipFile(kmz_file, 'r') as z:
        kml_filename = [f for f in z.namelist() if f.endswith('.kml')][0]
        with z.open(kml_filename) as f:
            return f.read()

def load_data(uploaded_file):
    if uploaded_file.name.endswith('.kmz'):
        kml_data = extract_kml_from_kmz(uploaded_file)
        temp_path = f"temp_{uploaded_file.name}.kml"
        with open(temp_path, "wb") as f:
            f.write(kml_data)
        gdf = gpd.read_file(temp_path, driver='KML')
        if os.path.exists(temp_path): os.remove(temp_path)
    else:
        gdf = gpd.read_file(uploaded_file, driver='KML')
    return gdf

st.set_page_config(page_title="ISP Network Planner Pro", layout="wide")
st.title("🌐 ISP Network Planner: Road-Based Analysis")
st.markdown("Fitur: **Jarak Jalan Max 250m**, **Naming by Pole Code**, & **Boundary Cluster**.")

col1, col2 = st.columns(2)
with col1:
    file_tiang = st.file_uploader("Upload KMZ Tiang", type=['kml', 'kmz'])
with col2:
    file_rumah = st.file_uploader("Upload KMZ Rumah", type=['kml', 'kmz'])

if file_tiang and file_rumah:
    try:
        gdf_tiang = load_data(file_tiang)
        gdf_rumah = load_data(file_rumah)
        st.success(f"Data Terbaca: {len(gdf_tiang)} Tiang & {len(gdf_rumah)} Rumah")

        if st.button("Jalankan Analisis Network"):
            with st.spinner("Mengunduh peta jalan dan menghitung rute..."):
                # Download Graph Jalan
                avg_lat = gdf_tiang.geometry.y.mean()
                avg_lon = gdf_tiang.geometry.x.mean()
                G = ox.graph_from_point((avg_lat, avg_lon), dist=2500, network_type='drive')
                
                kml_out = simplekml.Kml()
                colors = [simplekml.Color.red, simplekml.Color.blue, simplekml.Color.green, 
                          simplekml.Color.yellow, simplekml.Color.purple, simplekml.Color.orange]

                for i, tiang in gdf_tiang.iterrows():
                    kode_tiang = tiang.get('Name', f"Pole-{i+1}")
                    tiang_coord = (tiang.geometry.y, tiang.geometry.x)
                    tiang_node = ox.distance.nearest_nodes(G, tiang_coord[1], tiang_coord[0])
                    
                    fol = kml_out.newfolder(name=f"AREA_{kode_tiang}")
                    
                    # Titik Tiang
                    pt = fol.newpoint(name=f"PUSAT_{kode_tiang}", coords=[(tiang.geometry.x, tiang.geometry.y)])
                    pt.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/paddle/red-stars.png'

                    valid_homes = []
                    points_for_boundary = [(tiang.geometry.x, tiang.geometry.y)]

                    for j, rumah in gdf_rumah.iterrows():
                        rumah_coord = (rumah.geometry.y, rumah.geometry.x)
                        rumah_node = ox.distance.nearest_nodes(G, rumah_coord[1], rumah_coord[0])
                        
                        try:
                            dist = nx.shortest_path_length(G, tiang_node, rumah_node, weight='length')
                            if dist <= 250:
                                path = nx.shortest_path(G, tiang_node, rumah_node, weight='length')
                                line_coords = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in path]
                                valid_homes.append({
                                    'dist': dist, 'coords': (rumah.geometry.x, rumah.geometry.y),
                                    'path': line_coords, 'name': rumah.get('Name', f"H-{j}")
                                })
                        except: continue

                    valid_homes = sorted(valid_homes, key=lambda x: x['dist'])[:10]

                    for home in valid_homes:
                        points_for_boundary.append(home['coords'])
                        pr = fol.newpoint(name=f"{kode_tiang}-{home['name']}", coords=[home['coords']])
                        pr.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/paddle/wht-blank.png'
                        pr.style.iconstyle.scale = 0.6
                        
                        ls = fol.newlinestring(name=f"Route {home['name']}")
                        ls.coords = home['path']
                        ls.style.linestyle.width = 2
                        ls.style.linestyle.color = colors[i % len(colors)]

                    if len(points_for_boundary) >= 3:
                        pts_arr = np.array(points_for_boundary)
                        hull = ConvexHull(pts_arr)
                        hull_pts = np.vstack([pts_arr[hull.vertices], pts_arr[hull.vertices[0]]])
                        
                        poly = fol.newpolygon(name=f"BOUNDARY_{kode_tiang}")
                        poly.outerboundaryis = [(p[0], p[1]) for p in hull_pts]
                        poly.style.polystyle.color = simplekml.Color.changealphaint(50, colors[i % len(colors)])
                        poly.style.linestyle.color = colors[i % len(colors)]

                output_path = "ISP_Road_Analysis.kml"
                kml_out.save(output_path)
                st.success("Analisis Selesai!")
                with open(output_path, "rb") as f:
                    st.download_button("📥 Download KMZ Hasil", f, file_name="ISP_Final_Analysis.kmz")
    except Exception as e:
        st.error(f"Terjadi kesalahan teknis: {e}")