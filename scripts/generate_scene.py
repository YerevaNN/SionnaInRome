import os
import sys
import logging
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from src.scene_generation import (
    fetch_osm_buildings,
    convert_coordinates_to_local,
    create_building_ply,
    create_ground_plane_ply,
    generate_scene_xml
)
from config.config import SCENE_BOUNDS, ROME_CENTER, SCENES_DIR, MESHES_DIR

# logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__) 


def generate_rome_scene():
    """Generate Rome scene with proper building heights from OSM."""
    
    logger.info("Starting scene generation...")
    
    # Fetch buildings from OSM
    logger.info("Fetching buildings from OpenStreetMap...")
    buildings = fetch_osm_buildings(SCENE_BOUNDS)
    
    if not buildings:
        logger.error("No buildings fetched from OSM")
        return
    
    logger.info(f"Fetched {len(buildings)} buildings")
    
    # coordinates to local system
    logger.info("Converting coordinates to local system...")
    buildings_local = convert_coordinates_to_local(
        buildings, 
        ROME_CENTER["lat"], 
        ROME_CENTER["lon"]
    )
    
    # PLY files for buildings
    logger.info("Generating building meshes...")
    ply_files = []
    
    for building in buildings_local:
        building_id = building['id']
        ply_filename = f"{building_id}.ply"
        ply_path = MESHES_DIR / ply_filename
        
        if create_building_ply(building, str(ply_path)):
            ply_files.append(ply_filename)
            logger.debug(f"Created mesh: {ply_filename}")
        else:
            logger.warning(f"Failed to create mesh for {building_id}")
    
    logger.info(f"Created {len(ply_files)} building meshes")
    
    #  ground plane
    logger.info("Generating ground plane...")
    ground_path = MESHES_DIR / "Plane.ply"
    if create_ground_plane_ply(SCENE_BOUNDS, str(ground_path), margin_pct=0.1):
        ply_files.append("Plane.ply")
        logger.info("Created ground plane mesh")
    else:
        logger.error("Failed to create ground plane")
    
    #  XML scene file
    logger.info("Generating XML scene file...")
    xml_path = SCENES_DIR / "rome_scene_with_heights.xml"
    
    if generate_scene_xml(ply_files, str(xml_path)):
        logger.info(f"Successfully generated scene: {xml_path}")
        logger.info(f"Total meshes: {len(ply_files)}")
        logger.info(f"Meshes directory: {MESHES_DIR}")
    else:
        logger.error("Failed to generate XML scene")
    
    logger.info("Scene generation completed!")


if __name__ == "__main__":
    generate_rome_scene()