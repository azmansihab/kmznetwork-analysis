import streamlit as st
import geopandas as gpd
import pandas as pd
import osmnx as ox
import networkx as nx
import simplekml
import zipfile
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Point

# Konfigurasi OSMNX
ox.config(use_cache=True, log_console=False)

def extract_kml_from_kmz(kmz_file):
    with zipfile.ZipFile(kmz_file, 'r') as z:
        kml_filename = [f for f in z.namelist() if f.endswith('.kml')][0]
        with z.open(kml_filename) as f:
            return f.read()

def load_data(uploaded_file):
    # Geopandas membaca file KML/KMZ dan mengambil kolom 'Name' secara otomatis
    if uploaded_file.name.endswith('.kmz'):
        kml_data = extract_kml_from_kmz(uploaded_file)
        with open("temp.kml", "wb") as f: f.write(kml_data)
        gdf = gpd.read_file("temp.kml", driver='KML')
    else:
        gdf = gpd.read_file(uploaded_file, driver='KML')
    return gdf

st.set_page_config(page_title="ISP Network Analysis Pro", layout="wide")
st.title("🌐 Network Planner: Berbasis Kode Tiang & Jalan")
st.markdown("Fitur: **Jarak Jalan Max 250m**, **Nama Otomatis Sesuai Kode Tiang**, & **Boundary**.")

file_tiang = st.file_uploader("Upload KMZ/KML Tiang (Pusat)", type=['kml', 'kmz'])
file_rumah = st.file_uploader("Upload KMZ/KML Rumah (Target)", type=['kml', 'kmz'])

if file_tiang and file_rumah:
    gdf_tiang = load_data(file_tiang)
    gdf_rumah = load_data(file_rumah)
    
    st.info(f"Data Terbaca: {len(gdf_tiang)} Tiang. Pastikan kolom 'Name' di file Anda berisi kode tiang.")

    if st.button("Jalankan Analisis Network"):
        with st.spinner("Sedang menghitung rute jalan dan membuat boundary..."):
            # Ambil pusat area untuk download peta jalan OSM
            avg_lat, avg_lon = gdf_tiang.geometry.y.mean(), gdf_tiang.geometry.x.mean()
            G = ox.graph_from_point((avg_lat, avg_lon), dist=2500, network_type='drive')
            
            kml_out = simplekml.Kml()
            
            # Daftar warna untuk variasi boundary
            colors = [simplekml.Color.red, simplekml.Color.blue, simplekml.Color.green, 
                      simplekml.Color.yellow, simplekml.Color.purple, simplekml.Color.orange]

            for i, tiang in gdf_tiang.iterrows():
                # AMBIL KODE TIANG DARI INPUT
                kode_tiang = tiang.get('Name', f"Tiang-{i+1}")
                tiang_coord = (tiang.geometry.y, tiang.geometry.x)
                tiang_node = ox.distance.nearest_nodes(G, tiang_coord[1], tiang_coord[0])
                
                # Buat Folder Berdasarkan Kode Tiang
                fol = kml_out.newfolder(name=f"AREA_{kode_tiang}")
                
                # Titik Tiang Utama
                pt = fol.newpoint(name=f"PUSAT_{kode_tiang}", coords=[(tiang.geometry.x, tiang.geometry.y)])
                pt.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/paddle/red-stars.png'

                valid_homes = []
                points_for_boundary = [(tiang.geometry.x, tiang.geometry.y)]

                for j, rumah in gdf_rumah.iterrows():
                    rumah_coord = (rumah.geometry.y, rumah.geometry.x)
                    rumah_node = ox.distance.nearest_nodes(G, rumah_coord[1], rumah_coord[0])
                    
                    try:
                        dist = nx.shortest_path_length(G, tiang_node, rumah_node, weight='length')
                        if dist <= 250: # Rule Maksimal 250m
                            path = nx.shortest_path(G, tiang_node, rumah_node, weight='length')
                            line_coords = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in path]
                            
                            valid_homes.append({
                                'dist': dist,
                                'coords': (rumah.geometry.x, rumah.geometry.y),
                                'path': line_coords,
                                'name': rumah.get('Name', f"R-{j}")
                            })
                    except: continue

                # Ambil 10 terdekat
                valid_homes = sorted(valid_homes, key=lambda x: x['dist'])[:10]

                # Gambar Rute & Titik Rumah
                for home in valid_homes:
                    points_for_boundary.append(home['coords'])
                    # Titik Rumah
                    pr = fol.newpoint(name=f"{kode_tiang}-{home['name']}", coords=[home['coords']])
                    pr.style.iconstyle.scale = 0.7
                    # Garis Rute Jalan
                    ls = fol.newlinestring(name=f"Jalur {kode_tiang} ke {home['name']}")
                    ls.coords = home['path']
                    ls.style.linestyle.width = 2
                    ls.style.linestyle.color = colors[i % len(colors)]

                # BUAT BOUNDARY SESUAI KODE TIANG
                if len(points_for_boundary) >= 3:
                    pts_arr = np.array(points_for_boundary)
                    hull = ConvexHull(pts_arr)
                    hull_pts = pts_arr[hull.vertices]
                    hull_pts = np.vstack([hull_pts, hull_pts[0]]) # Tutup polygon
                    
                    poly = fol.newpolygon(name=f"BOUNDARY_{kode_tiang}")
                    poly.outerboundaryis = [(p[0], p[1]) for p in hull_pts]
                    # Styling Boundary
                    poly.style.polystyle.color = simplekml.Color.changealphaint(40, colors[i % len(colors)])
                    poly.style.linestyle.color = colors[i % len(colors)]
                    poly.style.linestyle.width = 3

            output_name = "Analisis_ISP_Final.kml"
            kml_out.save(output_name)
            st.success(f"Analisis Selesai untuk {len(gdf_tiang)} Tiang!")
            
            with open(output_name, "rb") as f:
                st.download_button(f"📥 Download Hasil KMZ ({output_name})", f, file_name=output_name)

except Exception as e:
    st.error(f"Error: {e}")