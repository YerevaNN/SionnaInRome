import os
import numpy as np
import math
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import re
import random
import osmnx as ox
from plyfile import PlyData, PlyElement
from shapely.geometry import Polygon, MultiPolygon
from pyproj import Transformer

logger = logging.getLogger(__name__)

# Configure OSMnx
ox.settings.log_console = False
ox.settings.use_cache = True
ox.settings.cache_folder = Path(__file__).parent.parent / "cache"
ox.settings.max_query_area_size = float('inf')
# Coordinate transformer
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)


def extract_height_from_osm(building):
    def parse_height_string(height_str):
        if not height_str:
            return None
        # Convert to string and clean
        height_str = str(height_str).strip().lower()
        # Extract number and unit using regex
        match = re.match(r'^(\d*\.?\d+)\s*(m|meters?|ft|feet|\')?', height_str)
        if match:
            value = float(match.group(1))
            unit = match.group(2) or 'm'  # Default to meters
            # Convert feet to meters
            if unit in ['ft', 'feet', "'"]:
                value = value * 0.3048
            return value
        return None
    
    # Method 1: Direct height attribute
    for height_key in ['height', 'building:height', 'roof:height']:
        if height_key in building and building[height_key]:
            height = parse_height_string(building[height_key])
            if height and 2.0 <= height <= 200.0:  # Reasonable bounds
                return height
    
    # Method 2: Building levels
    for levels_key in ['building:levels', 'levels']:
        if levels_key in building and building[levels_key]:
            try:
                levels = float(building[levels_key])
                if 1 <= levels <= 50:  # Reasonable bounds
                    # Vary floor height by building type
                    building_type = building.get('building', 'yes')
                    if building_type in ['industrial', 'warehouse']:
                        floor_height = 4.5  # Higher ceilings
                    elif building_type in ['commercial', 'office', 'retail']:
                        floor_height = 3.5  # Standard commercial
                    else:
                        floor_height = 3.0  # Residential
                    
                    return levels * floor_height
            except (ValueError, TypeError):
                pass
    
    # Method 3: Building type-specific defaults with variation
    building_type = building.get('building', 'yes')
    amenity = building.get('amenity', '')
    
    # Create realistic height ranges by type
    if building_type in ['house', 'residential', 'detached', 'semi_detached']:
        base_height = random.uniform(5.5, 8.5)
    elif building_type in ['apartments', 'dormitory']:
        base_height = random.uniform(12.0, 25.0)
    elif building_type in ['commercial', 'office', 'retail']:
        base_height = random.uniform(10.0, 20.0)
    elif building_type in ['industrial', 'warehouse', 'manufacture']:
        base_height = random.uniform(6.0, 12.0)
    elif building_type in ['church', 'cathedral', 'chapel']:
        base_height = random.uniform(15.0, 35.0)
    elif building_type in ['school', 'university']:
        base_height = random.uniform(8.0, 15.0)
    elif building_type in ['hospital']:
        base_height = random.uniform(15.0, 30.0)
    elif amenity in ['place_of_worship']:
        base_height = random.uniform(12.0, 25.0)
    elif amenity in ['school', 'university']:
        base_height = random.uniform(8.0, 15.0)
    else:
        # Generic building - vary by footprint size if available
        try:
            area = building.get('geometry', {}).area * 111320 * 111320  # Rough area in m²
            if area > 1000:  # Large building
                base_height = random.uniform(12.0, 25.0)
            elif area > 200:  # Medium building  
                base_height = random.uniform(8.0, 18.0)
            else:  # Small building
                base_height = random.uniform(5.0, 12.0)
        except:
            base_height = random.uniform(8.0, 15.0)
    
    return round(base_height, 1)


def fetch_osm_buildings(bounds: Dict[str, Tuple[float, float]]) -> List[Dict]:
    north = bounds['lat_range'][1]
    south = bounds['lat_range'][0]
    east = bounds['lon_range'][1]
    west = bounds['lon_range'][0]
    
    logger.info(f"Fetching OSM buildings for bounds: N={north}, S={south}, E={east}, W={west}")
    
    try:
        # Fetch buildings from OSM
        buildings = ox.features_from_bbox(bbox=(west, south, east, north), tags={'building': True})
        logger.info(f"Found {len(buildings)} buildings")
        
        # Check if any buildings were found
        if buildings.empty:
            logger.warning("No buildings found in the specified bounds")
            return []
        
        building_list = []
        for idx, (_, building) in enumerate(buildings.iterrows()):
            geometry = building.geometry
            
            # Handle different geometry types
            if isinstance(geometry, Polygon):
                polygons = [geometry]
            elif isinstance(geometry, MultiPolygon):
                polygons = list(geometry.geoms)
            else:
                continue
            
            # Extract building attributes
            building_attrs = {
                'id': f'building_{idx}',
                'building': building.get('building', 'yes'),
                'height': building.get('height'),
                'building:height': building.get('building:height'),
                'roof:height': building.get('roof:height'),
                'building:levels': building.get('building:levels'),
                'levels': building.get('levels'),
                'amenity': building.get('amenity'),
                'geometry': geometry,
            }
            
            # Get height using our extraction function
            height = extract_height_from_osm(building_attrs)
            
            # Process each polygon
            for poly_idx, polygon in enumerate(polygons):
                exterior_coords = list(polygon.exterior.coords)
                
                building_data = {
                    'id': f"{building_attrs['id']}_{poly_idx}" if len(polygons) > 1 else building_attrs['id'],
                    'exterior': exterior_coords,
                    'height': height,
                    'attributes': building_attrs,
                }
                
                building_list.append(building_data)
        
        logger.info(f"Fetched {len(building_list)} building polygons from OSM")
        return building_list
        
    except Exception as e:
        logger.error(f"Error fetching OSM buildings: {e}")
        return []


def convert_coordinates_to_local(buildings: List[Dict], center_lat: float, center_lon: float) -> List[Dict]:
    for building in buildings:
        exterior_local = []
        for lon, lat in building['exterior']:
            # Convert to local coordinates
            x, y = transformer.transform(lon, lat)
            center_x, center_y = transformer.transform(center_lon, center_lat)
            local_x = x - center_x
            local_y = y - center_y
            exterior_local.append((local_x, local_y))
        
        building['exterior'] = exterior_local
    
    return buildings


def triangulate_polygon(polygon_coords):
    triangles = []
    
    # Make sure we remove the last point if it's the same as the first
    coords = polygon_coords.copy()
    if len(coords) > 1 and coords[0] == coords[-1]:
        coords.pop()
    
    n = len(coords)
    
    if n < 3:
        return triangles  # Not enough points to form a triangle
    
    # Create triangles by connecting first point to pairs of consecutive vertices
    for i in range(1, n - 1):
        try:
            # Validate that all coordinates can be converted to float
            p0 = (float(coords[0][0]), float(coords[0][1]), 0.0)
            p1 = (float(coords[i][0]), float(coords[i][1]), 0.0)
            p2 = (float(coords[i+1][0]), float(coords[i+1][1]), 0.0)
            
            # Check for NaN or infinite values
            if (math.isnan(p0[0]) or math.isnan(p0[1]) or
                math.isnan(p1[0]) or math.isnan(p1[1]) or
                math.isnan(p2[0]) or math.isnan(p2[1]) or
                math.isinf(p0[0]) or math.isinf(p0[1]) or
                math.isinf(p1[0]) or math.isinf(p1[1]) or
                math.isinf(p2[0]) or math.isinf(p2[1])):
                continue
            
            triangles.append([p0, p1, p2])
        except (ValueError, TypeError):
            # Skip this triangle if coordinates can't be converted to float
            continue
    
    return triangles


def create_building_ply(building: Dict, output_path: str) -> bool:
    try:
        exterior = building['exterior']
        height = building.get('height', 10.0)
        
        if len(exterior) < 3:
            logger.warning(f"Building has insufficient vertices: {len(exterior)}")
            return False
        
        # Create mesh (ground, roof, walls)
        mesh = create_building_mesh(building)
        
        # Collect all vertices and faces
        vertices = []
        faces = []
        vertex_count = 0
        
        # Process each section (walls, roof, ground)
        for section in ['walls', 'roof', 'ground']:
            for triangle in mesh[section]:
                # Validate and clean vertices before adding
                valid_vertices = []
                for vertex in triangle:
                    try:
                        # Ensure each coordinate is a valid float
                        x = float(vertex[0])
                        y = float(vertex[1])
                        z = float(vertex[2])
                        
                        # Check for NaN or infinite values
                        if (math.isnan(x) or math.isnan(y) or math.isnan(z) or
                            math.isinf(x) or math.isinf(y) or math.isinf(z)):
                            continue
                        
                        valid_vertices.append((x, y, z))
                    except (ValueError, TypeError):
                        continue
                
                # Only add face if we have exactly 3 valid vertices
                if len(valid_vertices) == 3:
                    # Add vertices
                    for v in valid_vertices:
                        vertices.append(v)
                    
                    # Add face (triangle)
                    faces.append((np.array([vertex_count, vertex_count+1, vertex_count+2], dtype='i4'),))
                    vertex_count += 3
        
        # Skip if no valid vertices
        if len(vertices) == 0:
            logger.warning(f"No valid vertices for building")
            return False
        
        # Create PLY data structure
        vertex_data = np.array(vertices, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        vertex_element = PlyElement.describe(vertex_data, 'vertex')
        
        face_data = np.array(faces, dtype=[('vertex_indices', 'i4', (3,))])
        face_element = PlyElement.describe(face_data, 'face')
        
        # Create PLY data and save to file
        ply_data = PlyData([vertex_element, face_element], text=True)
        ply_data.write(output_path)
        
        return True
        
    except Exception as e:
        logger.error(f"Error creating PLY file {output_path}: {e}")
        return False


def create_building_mesh(building):
    exterior = building['exterior']
    try:
        height = float(building['height'])
        # Ensure height is valid
        if math.isnan(height) or math.isinf(height) or height <= 0:
            height = 10.0  # Use default if invalid
    except (ValueError, TypeError):
        height = 10.0  # Use default if conversion fails
    
    # Remove the last point if it's the same as the first (closed polygon)
    if len(exterior) > 1 and exterior[0] == exterior[-1]:
        exterior = exterior[:-1]
    
    # Ensure we have at least 3 points to form a polygon
    if len(exterior) < 3:
        return {'ground': [], 'roof': [], 'walls': []}
    
    # Ground face triangulation - explicitly add z=0 to make 3D points
    ground_triangles = triangulate_polygon(exterior)
    
    # Roof face triangulation (same as ground but elevated)
    roof_triangles = []
    for triangle in ground_triangles:
        try:
            roof_triangle = [
                (float(p[0]), float(p[1]), height) for p in triangle
            ]
            # Verify all points are valid
            valid = True
            for p in roof_triangle:
                if (math.isnan(p[0]) or math.isnan(p[1]) or math.isnan(p[2]) or
                    math.isinf(p[0]) or math.isinf(p[1]) or math.isinf(p[2])):
                    valid = False
                    break
            if valid:
                roof_triangles.append(roof_triangle)
        except (ValueError, TypeError):
            continue
    
    # Wall faces (connect ground and roof edges)
    wall_triangles = []
    for i in range(len(exterior)):
        try:
            p1 = exterior[i]
            p2 = exterior[(i+1) % len(exterior)]
            
            # Create 2 triangles for each rectangular wall face
            # Explicitly add z coordinates to make 3D points
            g1 = (float(p1[0]), float(p1[1]), 0.0)
            g2 = (float(p2[0]), float(p2[1]), 0.0)
            r1 = (float(p1[0]), float(p1[1]), height)
            r2 = (float(p2[0]), float(p2[1]), height)
            
            # Verify coordinates
            all_points = [g1, g2, r1, r2]
            valid = True
            for p in all_points:
                if (math.isnan(p[0]) or math.isnan(p[1]) or math.isnan(p[2]) or
                    math.isinf(p[0]) or math.isinf(p[1]) or math.isinf(p[2])):
                    valid = False
                    break
            
            if valid:
                wall_triangles.append([g1, g2, r1])
                wall_triangles.append([g2, r2, r1])
        except (ValueError, TypeError):
            continue
    
    # Combine all triangles
    all_triangles = {
        'ground': ground_triangles,
        'roof': roof_triangles,
        'walls': wall_triangles
    }
    
    return all_triangles


def create_ground_plane_ply(bounds: Dict, output_path: str, margin_pct: float = 0.1) -> bool:
    try:
        # Get local coordinate bounds
        # This is a simplified version - in production you'd use proper coordinate conversion
        width = 1000  # Example width in meters
        height = 1000  # Example height in meters
        
        # Add margin
        expand_x = width * margin_pct
        expand_y = height * margin_pct
        
        # Define corners
        left = -width/2 - expand_x
        right = width/2 + expand_x
        top = -height/2 - expand_y
        bottom = height/2 + expand_y
        z_level = -0.5
        
        # Plane vertices (counter-clockwise order)
        vertices = [
            (top, left, z_level),
            (top, right, z_level),
            (bottom, right, z_level),
            (bottom, left, z_level),
        ]
        
        # Two triangles → one quad
        faces = [
            (np.array([0, 1, 2], dtype='i4'),),
            (np.array([0, 2, 3], dtype='i4'),),
        ]
        
        # Write PLY
        vertex_array = np.array(vertices,
                               dtype=[('x', 'f4'),
                                      ('y', 'f4'),
                                      ('z', 'f4')])
        face_array = np.array(faces,
                             dtype=[('vertex_indices', 'i4', (3,))])
        
        ply_data = PlyData([
            PlyElement.describe(vertex_array, 'vertex'),
            PlyElement.describe(face_array, 'face')
        ], text=True)
        
        ply_data.write(output_path)
        return True
        
    except Exception as e:
        logger.error(f"Error creating ground plane: {e}")
        return False


def prettify(elem):
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def generate_scene_xml(ply_files: List[str], output_path: str) -> bool:
    try:
        # Create the root element
        scene = ET.Element("scene", version="2.1.0")
        
        # Add a comment
        scene.append(ET.Comment(" Camera and Rendering Parameters "))
        
        # Add integrator
        integrator = ET.SubElement(scene, "integrator", type="path", id="elm__0", name="elm__0")
        ET.SubElement(integrator, "integer", name="max_depth", value="12")
        
        # Add materials section
        scene.append(ET.Comment(" Materials "))
        
        # Add concrete material
        bsdf = ET.SubElement(scene, "bsdf", type="twosided", id="mat-itu_concrete")
        bsdf_diffuse = ET.SubElement(bsdf, "bsdf", type="diffuse")
        ET.SubElement(bsdf_diffuse, "rgb", value="0.539479 0.539479 0.539480", name="reflectance")
        
        # Add emitters section
        scene.append(ET.Comment(" Emitters "))
        
        # Add shapes section
        scene.append(ET.Comment(" Shapes "))
        
        # Add ground plane reference
        if "Plane.ply" in ply_files:
            ground = ET.SubElement(scene, "shape", type="ply", id="elm__2", name="elm__2")
            ET.SubElement(ground, "string", name="filename", value="meshes/Plane.ply")
            ET.SubElement(ground, "ref", id="mat-itu_concrete", name="bsdf")
        
        # Add building references
        building_ply_files = [f for f in ply_files if f != 'Plane.ply' and f is not None]
        for i, building_file in enumerate(building_ply_files):
            building = ET.SubElement(scene, "shape", type="ply",
                                   id=f"elm__{i+3}", name=f"elm__{i+3}")
            ET.SubElement(building, "string", name="filename",
                         value=f"meshes/{building_file}")
            ET.SubElement(building, "ref", id="mat-itu_concrete", name="bsdf")
        
        # Add volumes section
        scene.append(ET.Comment(" Volumes "))
        
        # Write XML file
        pretty_xml = prettify(scene)
        
        with open(output_path, 'w') as f:
            f.write(pretty_xml)
        
        logger.info(f"Generated XML scene: {output_path}")
        logger.info(f"Added {len(building_ply_files)} building shapes to scene")
        
        return True
        
    except Exception as e:
        logger.error(f"Error generating XML: {e}")
        return False