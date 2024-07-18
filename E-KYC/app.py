import os
import logging
import streamlit as st
from sqlalchemy import create_engine
from preprocess import read_image, extract_id_card, save_image
from ocr_engine import extract_text
from postprocess import extract_information
from face_verification import detect_and_extract_face, face_comparison, get_face_embeddings
from mysqldb_operations import insert_records, fetch_records, check_duplicacy
import yaml

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir,"ekyc_logs.log"), level=logging.INFO, format=logging_str, filemode="a")

def read_yaml(path_to_yaml):
    with open(path_to_yaml, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

# Read database configuration from YAML file
config_path = "config.yaml"
try:
    config = read_yaml(config_path)
    db_config = config['database']
    username = db_config['username']
    password = db_config['password']
    host = db_config['host']
    port = db_config['port']
    database = db_config['database']
    dialect = db_config['dialect']

    # Create a connection string
    connection_string = f"{dialect}://{username}:{password}@{host}:{port}/{database}"

    # Create the database engine
    engine = create_engine(connection_string)
except FileNotFoundError:
    st.error("Configuration file not found. Please ensure 'config.yaml' is present in the same directory.")
    logging.error("Configuration file not found.")
except KeyError as e:
    st.error(f"Configuration file is missing a required key: {e}")
    logging.error(f"Configuration file is missing a required key: {e}")

def wider_page():
    max_width_str = "max-width: 1200px;"
    st.markdown(
        f"""
        <style>
            .reportview-container .main .block-container{{ {max_width_str} }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    logging.info("Page layout set to wider configuration.")

def set_custom_theme():
    st.markdown(
        """
        <style>
            body {
                background-color: #f0f2f6; /* Set background color */
                color: #333333; /* Set text color */
            }
            .sidebar .sidebar-content {
                background-color: #ffffff; /* Set sidebar background color */
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    logging.info("Custom theme applied to Streamlit app.")

def sidebar_section():
    st.sidebar.title("Select ID Card Type")
    option = st.sidebar.selectbox("ID Card Type", ("PAN", "Aadhar"))
    logging.info(f"ID card type selected: {option}")
    return option

def header_section(option):
    if option == "Aadhar":
        st.title("Registration Using Aadhar Card")
        logging.info("Header set for Aadhar Card registration.")
    elif option == "PAN":
        st.title("Registration Using PAN Card")
        logging.info("Header set for PAN Card registration.")

def main_content(image_file, face_image_file, engine):
    if image_file is not None:
        face_image = read_image(face_image_file, is_uploaded=True)
        if face_image is None:
            st.error("Face image not uploaded or could not be read. Please upload a valid face image.")
            logging.error("Face image not uploaded or could not be read.")
            return

        image = read_image(image_file, is_uploaded=True)
        if image is None:
            st.error("ID card image not uploaded or could not be read. Please upload a valid ID card image.")
            logging.error("ID card image not uploaded or could not be read.")
            return

        logging.info("ID card image and face image loaded.")
        
        image_roi, _ = extract_id_card(image)
        logging.info("ID card ROI extracted.")

        face_image_path2 = detect_and_extract_face(image_roi)
        if face_image_path2 is None:
            st.error("Failed to extract face from ID card image.")
            logging.error("Failed to extract face from ID card image.")
            return

        face_image_path1 = save_image(face_image, "face_image.jpg", path="data\\02_intermediate_data")
        logging.info("Faces extracted and saved.")
        
        is_face_verified = face_comparison(image1_path=face_image_path1, image2_path=face_image_path2)
        logging.info(f"Face verification status: {'successful' if is_face_verified else 'failed'}.")

        if is_face_verified:
            extracted_text = extract_text(image_roi)
            text_info = extract_information(extracted_text)
            logging.info("Text extracted and information parsed from ID card.")
            records = fetch_records(text_info, engine)
            if records.shape[0] > 0:
                st.write(records.shape)
                st.write(records)
            is_duplicate = check_duplicacy(text_info, engine)
            if is_duplicate:
                st.write(f"User already present with ID {text_info['ID']}")
            else: 
                st.write(text_info)
                text_info['DOB'] = text_info['DOB'].strftime('%Y-%m-%d')
                text_info['Embedding'] = get_face_embeddings(face_image_path1)
                insert_records(text_info, engine)
                logging.info(f"New user record inserted: {text_info['ID']}")
                
        else:
            st.error("Face verification failed. Please try again.")

    else:
        st.warning("Please upload an ID card image.")
        logging.warning("No ID card image uploaded.")

def main():
    wider_page()
    set_custom_theme()
    option = sidebar_section()
    header_section(option)
    image_file = st.file_uploader("Upload ID Card")
    if image_file is not None:
        face_image_file = st.file_uploader("Upload Face Image")
        main_content(image_file, face_image_file, engine)

if __name__ == "__main__":
    main()
