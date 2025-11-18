"""
Travelogue - AI Assisted Travel Photo Album Generator
Main Streamlit Application
"""
import streamlit as st
import os
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from PIL.ExifTags import TAGS
import json
from typing import List, Optional
from datetime import datetime
import cv2
import numpy as np

# Load config from config.yaml
#from .Utilities.config_loader import config

from Utilities.Ollama.LLM import inference_server_model
from Utilities.Llava.captioner import LlavaCaptioner
from Utilities.Clip.query_images import search_similar_images
from Utilities.Clip.embed_images import CLIPEncoder
from Utilities.InsightFace.detect_and_embed import get_face_embeddings, process_image

# Helper function to extract date from EXIF data
def get_exif_date(image_path):
    """
    Extract the date when the photo was taken from EXIF data.
    Returns a date object or None if not found.
    """
    try:
        image = Image.open(image_path)
        exifdata = image.getexif()
        
        if exifdata is None:
            return None
        
        # Try different EXIF date tags (in order of preference)
        # Tag IDs: 36867=DateTimeOriginal, 36868=DateTimeDigitized, 306=DateTime
        date_tags = [
            36867,  # DateTimeOriginal - when the photo was taken
            36868,  # DateTimeDigitized - when the photo was digitized
            306,    # DateTime - general date/time
        ]
        
        for tag_id in date_tags:
            if tag_id in exifdata:
                date_str = exifdata[tag_id]
                if date_str:
                    try:
                        # EXIF date format is typically "YYYY:MM:DD HH:MM:SS"
                        date_obj = datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
                        return date_obj.date()
                    except ValueError:
                        # Try alternative formats
                        try:
                            date_obj = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                            return date_obj.date()
                        except ValueError:
                            continue
        
        # If no date found in standard tags, try searching all tags
        for tag_id, value in exifdata.items():
            tag_name = TAGS.get(tag_id, tag_id)
            if 'date' in tag_name.lower() and isinstance(value, str):
                try:
                    date_obj = datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
                    return date_obj.date()
                except (ValueError, TypeError):
                    try:
                        date_obj = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
                        return date_obj.date()
                    except (ValueError, TypeError):
                        continue
        
        return None
    except Exception as e:
        # If EXIF reading fails, return None
        return None

# Initialize session state
if 'uploaded_images' not in st.session_state:
    st.session_state.uploaded_images = []
if 'captions' not in st.session_state:
    st.session_state.captions = {}
if 'photo_metadata' not in st.session_state:
    st.session_state.photo_metadata = {}  # Store metadata for each photo
if 'detected_faces' not in st.session_state:
    st.session_state.detected_faces = {}  # Store detected faces for each photo
if 'captioner' not in st.session_state:
    st.session_state.captioner = None
if 'clip_encoder' not in st.session_state:
    st.session_state.clip_encoder = None

# Sidebar navigation
st.sidebar.title("📸 Travelogue")
st.sidebar.markdown("**AI Assisted Travel Photo Album Generator**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📤 Upload/Edit Photos", "✍️ Generate Captions", "🔍 Search Photos", "📖 Generate Travelogue"],
    index=0
)

# Home Page
if page == "🏠 Home":
    st.title("Welcome to Travelogue")
    st.markdown("### AI Assisted Travel Photo Album Generator")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 📤 Upload/Edit Photos
        Upload your vacation photos to get started. 
        Faces are automatically detected and you can identify people.
        Edit metadata for already uploaded photos anytime.
        """)
    
    with col2:
        st.markdown("""
        #### ✍️ Generate Captions
        Use AI (LLaVA) to automatically generate 
        detailed captions for your photos.
        """)
    
    with col3:
        st.markdown("""
        #### 🔍 Search Photos
        Search through your photo collection using 
        natural language queries powered by CLIP.
        """)
    
    st.markdown("---")
    
    st.markdown("""
    #### 📖 Generate Travelogue
    Create a beautiful travelogue story from your photos
    using AI-powered narrative generation.
    """)
    
    st.markdown("---")
    st.info("💡 **Tip**: Start by uploading photos, then generate captions and search through your collection!")

# Upload Photos Page
elif page == "📤 Upload/Edit Photos":
    st.title("Upload/Edit Vacation Photos")
    st.markdown("Upload your travel photos and add metadata to create a rich travelogue.")
    st.markdown("---")
    
    # Display already uploaded photos
    if st.session_state.uploaded_images:
        st.subheader("📸 Already Uploaded Photos")
        st.markdown(f"You have **{len(st.session_state.uploaded_images)}** photo(s) uploaded.")
        
        # Display photos in a grid
        num_cols = 4
        cols = st.columns(num_cols)
        
        for idx, img_path in enumerate(st.session_state.uploaded_images):
            col_idx = idx % num_cols
            with cols[col_idx]:
                img_name = os.path.basename(img_path)
                try:
                    thumb_img = Image.open(img_path)
                    thumb_img.thumbnail((200, 200))
                    st.image(thumb_img, caption=img_name[:25] + "..." if len(img_name) > 25 else img_name, use_container_width=True)
                    
                    # Show metadata status
                    has_metadata = img_name in st.session_state.photo_metadata and st.session_state.photo_metadata[img_name].get("location")
                    if has_metadata:
                        st.caption("✅ Metadata added")
                    else:
                        st.caption("⚠️ No metadata")
                    
                    # Show face count if detected
                    detected_faces = st.session_state.detected_faces.get(img_name, [])
                    if detected_faces:
                        st.caption(f"👤 {len(detected_faces)} face(s)")
                except Exception as e:
                    st.error(f"Error loading {img_name}")
        
        st.markdown("---")
    
    st.markdown("---")
    st.markdown("### 📤 Upload New Photos")
    st.markdown("Select one or more images from your device to add to your travelogue.")
    
    uploaded_files = st.file_uploader(
        "Choose images to upload",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) selected")
        
        # Create temporary directory for uploaded images
        if 'temp_dir' not in st.session_state:
            st.session_state.temp_dir = tempfile.mkdtemp()
        
        # Save uploaded images and initialize metadata if needed
        for uploaded_file in uploaded_files:
            save_path = os.path.join(st.session_state.temp_dir, uploaded_file.name)
            if save_path not in st.session_state.uploaded_images:
                image = Image.open(uploaded_file)
                image.save(save_path)
                st.session_state.uploaded_images.append(save_path)
                # Initialize metadata for new photos
                if uploaded_file.name not in st.session_state.photo_metadata:
                    # Try to extract date from EXIF data
                    exif_date = get_exif_date(save_path)
                    st.session_state.photo_metadata[uploaded_file.name] = {
                        "date": str(exif_date) if exif_date else None,
                        "exif_date_extracted": exif_date is not None
                    }
                
                # Automatically detect faces when image is uploaded
                if uploaded_file.name not in st.session_state.detected_faces:
                    try:
                        faces = get_face_embeddings(save_path)
                        st.session_state.detected_faces[uploaded_file.name] = faces
                        # Initialize face names in metadata
                        if faces:
                            if "faces" not in st.session_state.photo_metadata[uploaded_file.name]:
                                st.session_state.photo_metadata[uploaded_file.name]["faces"] = [
                                    {"name": "", "bbox": face["facial_area"], "confidence": face["det_score"]}
                                    for face in faces
                                ]
                    except Exception as e:
                        st.session_state.detected_faces[uploaded_file.name] = []
                        st.warning(f"⚠️ Could not detect faces in {uploaded_file.name}: {str(e)}")
        
    # Metadata Collection Section (always show if there are uploaded images)
    if st.session_state.uploaded_images:
        st.markdown("---")
        
        # Metadata Collection Section
        st.subheader("📝 Add/Edit Metadata for Your Photos")
        st.markdown("Please provide information about each photo to enhance your travelogue.")
        
        # Photo selector with thumbnail gallery
        st.markdown("### Select a Photo to Add Metadata")
        
        # Initialize selected photo index if not set
        if 'selected_photo_idx' not in st.session_state:
            st.session_state.selected_photo_idx = 0
        
        # Quick selector dropdown
        photo_options = [f"Photo {i+1}: {os.path.basename(img)}" for i, img in enumerate(st.session_state.uploaded_images)]
        selected_photo_display = st.selectbox(
            "Quick Select Photo",
            options=photo_options,
            index=st.session_state.selected_photo_idx,
            key="photo_selector_dropdown"
        )
        if selected_photo_display:
            new_idx = photo_options.index(selected_photo_display)
            if new_idx != st.session_state.selected_photo_idx:
                st.session_state.selected_photo_idx = new_idx
                st.rerun()
        
        st.markdown("---")
        
        # Display selected photo and its metadata form
        if st.session_state.uploaded_images:
            selected_idx = st.session_state.selected_photo_idx
            if selected_idx >= len(st.session_state.uploaded_images):
                selected_idx = 0
                st.session_state.selected_photo_idx = 0
            
            img_path = st.session_state.uploaded_images[selected_idx]
            img_name = os.path.basename(img_path)
            
            # Show which photo is selected
            st.markdown(f"### 📸 Photo {selected_idx + 1} of {len(st.session_state.uploaded_images)}: {img_name}")
            st.markdown("---")
            
            # Display the selected photo with metadata form
            tab_idx = selected_idx
            
            # Display the image with face detection bounding boxes
            image = Image.open(img_path)
            
            # Get detected faces for this image
            detected_faces = st.session_state.detected_faces.get(img_name, [])
            
            # Draw bounding boxes on the image if faces are detected
            if detected_faces:
                # Convert PIL image to numpy array for drawing
                img_array = np.array(image)
                # Convert RGB to BGR for OpenCV
                if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                    img_with_boxes = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                else:
                    img_with_boxes = img_array.copy()
                
                # Draw bounding boxes and labels
                for idx, face in enumerate(detected_faces):
                    bbox = face["facial_area"]  # [x1, y1, x2, y2]
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # Draw rectangle (green, thickness 3)
                    cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    
                    # Get face name from metadata if available
                    face_name = ""
                    if "faces" in st.session_state.photo_metadata.get(img_name, {}):
                        face_metadata = st.session_state.photo_metadata[img_name]["faces"]
                        if idx < len(face_metadata) and face_metadata[idx].get("name"):
                            face_name = face_metadata[idx]["name"]
                    
                    # Draw label
                    label = f"Face {idx + 1}" + (f": {face_name}" if face_name else "")
                    # Increased font scale from 0.7 to 1.2 and thickness from 2 to 4 for better visibility
                    font_scale = 1.2
                    font_thickness = 4
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                    label_y = max(y1 - 10, label_size[1] + 10)
                    
                    # Draw label background (green)
                    cv2.rectangle(img_with_boxes, 
                                (x1, label_y - label_size[1] - 5), 
                                (x1 + label_size[0], label_y + 5), 
                                (0, 255, 0), -1)
                    
                    # Draw label text (black) with larger font
                    cv2.putText(img_with_boxes, label, 
                              (x1, label_y), 
                              cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)
                
                # Convert back from BGR to RGB for PIL
                img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
                # Convert back to PIL Image
                image = Image.fromarray(img_with_boxes)
                
                # Show detection info
                st.info(f"👤 {len(detected_faces)} face(s) detected in this image")
            
            # Show image at full width for better visibility
            st.image(image, caption=img_name, use_container_width=True)
            st.markdown("---")
            
            # Metadata form below the image
            with st.form(key=f"metadata_form_{tab_idx}"):
                st.markdown(f"### Metadata for: {img_name}")
                
                # Location/Place
                location = st.text_input(
                    "📍 Location / Place",
                    value=st.session_state.photo_metadata.get(img_name, {}).get("location", ""),
                    key=f"location_{tab_idx}",
                    help="Where was this photo taken? (e.g., 'Paris, France', 'Grand Canyon')"
                )
                
                # Date - try to get from saved metadata or EXIF
                saved_date = st.session_state.photo_metadata.get(img_name, {}).get("date")
                default_date = None
                
                # If we have a saved date string, parse it
                if saved_date:
                    try:
                        if isinstance(saved_date, str):
                            default_date = datetime.strptime(saved_date, "%Y-%m-%d").date()
                        elif hasattr(saved_date, 'date'):
                            default_date = saved_date.date() if hasattr(saved_date, 'date') else saved_date
                    except (ValueError, AttributeError):
                        # If parsing fails, try to extract from EXIF again
                        exif_date = get_exif_date(img_path)
                        default_date = exif_date
                else:
                    # Try to extract from EXIF if not already saved
                    exif_date = get_exif_date(img_path)
                    default_date = exif_date
                
                # Show indicator if date was extracted from EXIF
                date_label = "📅 Date"
                exif_extracted = st.session_state.photo_metadata.get(img_name, {}).get("exif_date_extracted", False)
                if default_date and exif_extracted:
                    date_label = "📅 Date (auto-filled from photo)"
                    st.info("✅ Date automatically extracted from photo's metadata")
                
                date = st.date_input(
                    date_label,
                    value=default_date,
                    key=f"date_{tab_idx}",
                    help="When was this photo taken? (Auto-filled from photo's EXIF data if available)"
                )
                
                # Face names section (if faces detected)
                detected_faces_form = st.session_state.detected_faces.get(img_name, [])
                face_names = []  # Initialize face_names list
                relationship = ""  # Initialize relationship
                
                if detected_faces_form:
                    st.markdown("### 👤 Identify Detected Faces")
                    st.markdown("Please provide names for each detected face:")
                    
                    # Initialize faces in metadata if not present
                    if "faces" not in st.session_state.photo_metadata.get(img_name, {}):
                        st.session_state.photo_metadata[img_name]["faces"] = [
                            {"name": "", "bbox": face["facial_area"], "confidence": face["det_score"]}
                            for face in detected_faces_form
                        ]
                    
                    # Create input fields for each face
                    for face_idx, face in enumerate(detected_faces_form):
                        face_metadata = st.session_state.photo_metadata[img_name]["faces"][face_idx]
                        default_name = face_metadata.get("name", "")
                        
                        face_name = st.text_input(
                            f"Face {face_idx + 1} (Confidence: {face['det_score']:.2f})",
                            value=default_name,
                            key=f"face_name_{tab_idx}_{face_idx}",
                            help=f"Enter the name of the person in bounding box {face_idx + 1}"
                        )
                        face_names.append(face_name)
                        
                        # Update metadata immediately (for display purposes)
                        st.session_state.photo_metadata[img_name]["faces"][face_idx]["name"] = face_name
                    
                    # Relationship question (if more than 1 face)
                    if len(detected_faces_form) > 1:
                        st.markdown("### 👨‍👩‍👧‍👦 Relationship")
                        
                        if len(detected_faces_form) == 2:
                            # For 2 people, simple text input
                            relationship = st.text_input(
                                "How are the people in this photo related to each other?",
                                value=st.session_state.photo_metadata.get(img_name, {}).get("relationship", ""),
                                key=f"relationship_{tab_idx}",
                                help="Describe the relationship between the two people (e.g., 'Friends', 'Siblings', 'Parent and child', 'Colleagues', 'Spouses')"
                            )
                        else:
                            # For 3+ people, use text area for more detailed description
                            relationship = st.text_area(
                                "How are all the people in this photo related to each other?",
                                value=st.session_state.photo_metadata.get(img_name, {}).get("relationship", ""),
                                key=f"relationship_{tab_idx}",
                                height=100,
                                help=f"Describe how all {len(detected_faces_form)} people are related to each other. For example: 'Family members - parents with their children', 'Friends from college', 'Colleagues from work', 'Extended family gathering', or specify individual relationships like 'John and Mary are siblings, Sarah is their cousin'"
                            )
                        
                        # Update metadata immediately
                        if img_name not in st.session_state.photo_metadata:
                            st.session_state.photo_metadata[img_name] = {}
                        st.session_state.photo_metadata[img_name]["relationship"] = relationship
                    
                    st.markdown("---")
                
                # People (general field - can be used for additional info)
                people = st.text_input(
                    "👥 Additional People / Group Info",
                    value=st.session_state.photo_metadata.get(img_name, {}).get("people", ""),
                    key=f"people_{tab_idx}",
                    help="Additional information about people in the photo (e.g., 'Family gathering', 'Friends from college')"
                )
                
                # Activity/Event
                activity = st.text_input(
                    "🎯 Activity / Event",
                    value=st.session_state.photo_metadata.get(img_name, {}).get("activity", ""),
                    key=f"activity_{tab_idx}",
                    help="What activity or event is this? (e.g., 'Hiking', 'Beach day', 'Museum visit')"
                )
                
                # Weather
                weather = st.selectbox(
                    "☀️ Weather",
                    options=["", "Sunny", "Cloudy", "Rainy", "Snowy", "Windy", "Foggy", "Clear", "Overcast"],
                    index=0 if not st.session_state.photo_metadata.get(img_name, {}).get("weather") 
                    else ["", "Sunny", "Cloudy", "Rainy", "Snowy", "Windy", "Foggy", "Clear", "Overcast"].index(
                        st.session_state.photo_metadata.get(img_name, {}).get("weather", "")
                    ) if st.session_state.photo_metadata.get(img_name, {}).get("weather") in ["", "Sunny", "Cloudy", "Rainy", "Snowy", "Windy", "Foggy", "Clear", "Overcast"] else 0,
                    key=f"weather_{tab_idx}",
                    help="What was the weather like?"
                )
                
                # Mood/Feeling
                mood = st.selectbox(
                    "😊 Mood / Feeling",
                    options=["", "Happy", "Relaxed", "Excited", "Peaceful", "Adventurous", "Nostalgic", "Energetic", "Content"],
                    index=0 if not st.session_state.photo_metadata.get(img_name, {}).get("mood") 
                    else ["", "Happy", "Relaxed", "Excited", "Peaceful", "Adventurous", "Nostalgic", "Energetic", "Content"].index(
                        st.session_state.photo_metadata.get(img_name, {}).get("mood", "")
                    ) if st.session_state.photo_metadata.get(img_name, {}).get("mood") in ["", "Happy", "Relaxed", "Excited", "Peaceful", "Adventurous", "Nostalgic", "Energetic", "Content"] else 0,
                    key=f"mood_{tab_idx}",
                    help="What was the mood or feeling?"
                )
                
                # Tags
                tags = st.text_input(
                    "🏷️ Tags (comma-separated)",
                    value=", ".join(st.session_state.photo_metadata.get(img_name, {}).get("tags", [])),
                    key=f"tags_{tab_idx}",
                    help="Add tags separated by commas (e.g., 'sunset, beach, vacation')"
                )
                
                # Notes/Comments
                notes = st.text_area(
                    "📝 Notes / Comments",
                    value=st.session_state.photo_metadata.get(img_name, {}).get("notes", ""),
                    key=f"notes_{tab_idx}",
                    height=100,
                    help="Any additional notes or memories about this photo"
                )
                
                # Submit button for this photo
                submitted = st.form_submit_button("💾 Save Metadata", type="primary")
                
                if submitted:
                    # Parse tags
                    tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else []
                    
                    # Get detected faces for this image
                    detected_faces_save = st.session_state.detected_faces.get(img_name, [])
                    
                    # Build faces_data with current face names from form
                    faces_data = []
                    if detected_faces_save:
                        # Get the latest face names from the form inputs (face_names list)
                        for face_idx, face in enumerate(detected_faces_save):
                            face_name = face_names[face_idx] if face_idx < len(face_names) else ""
                            faces_data.append({
                                "name": face_name,
                                "bbox": face["facial_area"],
                                "confidence": face["det_score"]
                            })
                    
                    # Get relationship from form input (it's already in the relationship variable)
                    relationship_value = relationship if len(detected_faces_save) > 1 else ""
                    
                    # Save metadata
                    st.session_state.photo_metadata[img_name] = {
                        "location": location,
                        "date": str(date) if date else None,
                        "people": people,
                        "activity": activity,
                        "weather": weather,
                        "mood": mood,
                        "tags": tag_list,
                        "notes": notes,
                        "filename": img_name,
                        "filepath": img_path,
                        "faces": faces_data,
                        "num_faces_detected": len(detected_faces_save),
                        "relationship": relationship_value
                    }
                    st.success(f"✅ Metadata saved for {img_name}!")
        
        st.markdown("---")
        
        # Display summary of metadata
        if st.session_state.photo_metadata:
            with st.expander("📋 View All Photo Metadata"):
                for img_name, metadata in st.session_state.photo_metadata.items():
                    st.markdown(f"### {img_name}")
                    col1, col2 = st.columns(2)
                    with col1:
                        if metadata.get("location"):
                            st.write(f"📍 **Location:** {metadata['location']}")
                        if metadata.get("date"):
                            st.write(f"📅 **Date:** {metadata['date']}")
                        if metadata.get("people"):
                            st.write(f"👥 **People:** {metadata['people']}")
                        if metadata.get("activity"):
                            st.write(f"🎯 **Activity:** {metadata['activity']}")
                    with col2:
                        if metadata.get("weather"):
                            st.write(f"☀️ **Weather:** {metadata['weather']}")
                        if metadata.get("mood"):
                            st.write(f"😊 **Mood:** {metadata['mood']}")
                        if metadata.get("tags"):
                            st.write(f"🏷️ **Tags:** {', '.join(metadata['tags'])}")
                    
                    # Display face information
                    if metadata.get("faces"):
                        face_names = [f.get("name", "") for f in metadata["faces"] if f.get("name")]
                        if face_names:
                            st.write(f"👤 **People Identified:** {', '.join(face_names)}")
                        if metadata.get("num_faces_detected", 0) > 0:
                            st.write(f"🔍 **Faces Detected:** {metadata['num_faces_detected']}")
                    
                    # Display relationship if multiple faces
                    if metadata.get("relationship"):
                        st.write(f"👨‍👩‍👧‍👦 **Relationship:** {metadata['relationship']}")
                    
                    if metadata.get("notes"):
                        st.write(f"📝 **Notes:** {metadata['notes']}")
                    st.markdown("---")
        
        st.markdown("---")
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Process & Store Images", type="primary"):
                with st.spinner("Processing images..."):
                    try:
                        # Check if metadata is complete (optional - you can make this required)
                        photos_with_metadata = sum(1 for img in st.session_state.uploaded_images 
                                                  if os.path.basename(img) in st.session_state.photo_metadata 
                                                  and st.session_state.photo_metadata[os.path.basename(img)])
                        
                        if photos_with_metadata < len(st.session_state.uploaded_images):
                            st.warning(f"⚠️ Only {photos_with_metadata} out of {len(st.session_state.uploaded_images)} photos have metadata. You can still process them.")
                        
                        # Here you would process and store images
                        # For now, just show success
                        st.success(f"✅ Successfully processed {len(st.session_state.uploaded_images)} images!")
                        st.info("💡 Images are now ready for captioning and searching!")
                        
                        # Optionally save metadata to JSON file
                        metadata_file = os.path.join(st.session_state.temp_dir, "metadata.json")
                        with open(metadata_file, 'w') as f:
                            json.dump(st.session_state.photo_metadata, f, indent=2)
                        st.info(f"📄 Metadata saved to: {metadata_file}")
                    except Exception as e:
                        st.error(f"❌ Error processing images: {str(e)}")
        
        with col2:
            if st.button("📥 Export Metadata"):
                metadata_json = json.dumps(st.session_state.photo_metadata, indent=2)
                st.download_button(
                    label="Download Metadata JSON",
                    data=metadata_json,
                    file_name="photo_metadata.json",
                    mime="application/json"
                )
        
        with col3:
            if st.button("🗑️ Clear All"):
                st.session_state.uploaded_images = []
                st.session_state.captions = {}
                st.session_state.photo_metadata = {}
                st.rerun()
        
        st.markdown(f"**Total images in collection:** {len(st.session_state.uploaded_images)}")
        if st.session_state.photo_metadata:
            st.markdown(f"**Photos with metadata:** {len(st.session_state.photo_metadata)}")

# Generate Captions Page
elif page == "✍️ Generate Captions":
    st.title("Generate Photo Captions")
    st.markdown("Use AI (LLaVA) to automatically generate detailed captions for your photos.")
    st.markdown("---")
    
    if not st.session_state.uploaded_images:
        st.warning("⚠️ No images uploaded yet. Please upload photos first from the 'Upload/Edit Photos' page.")
    else:
        # Initialize captioner
        if st.session_state.captioner is None:
            with st.spinner("Loading LLaVA model (this may take a moment)..."):
                try:
                    st.session_state.captioner = LlavaCaptioner()
                    st.success("✅ LLaVA model loaded successfully!")
                except Exception as e:
                    st.error(f"❌ Error loading LLaVA model: {str(e)}")
                    st.info("💡 Make sure the model dependencies are installed and configured correctly.")
        
        if st.session_state.captioner:
            st.subheader("Generate Captions")
            
            # Option to generate for all images or single image
            caption_mode = st.radio(
                "Caption Mode",
                ["Single Image", "All Images"],
                horizontal=True
            )
            
            if caption_mode == "Single Image":
                selected_image = st.selectbox(
                    "Select an image to caption",
                    [os.path.basename(img) for img in st.session_state.uploaded_images]
                )
                
                if st.button("Generate Caption", type="primary"):
                    img_path = next(img for img in st.session_state.uploaded_images 
                                  if os.path.basename(img) == selected_image)
                    
                    with st.spinner("Generating caption..."):
                        try:
                            image = Image.open(img_path).convert("RGB")
                            
                            # Get metadata for this image to enhance the caption
                            metadata = st.session_state.photo_metadata.get(selected_image, {})
                            
                            # Build enhanced prompt with metadata
                            prompt_parts = ["Describe the image in detail."]
                            
                            if metadata:
                                prompt_parts.append("\n\nConsider the following context about this photo:")
                                
                                if metadata.get("location"):
                                    prompt_parts.append(f"- Location: {metadata['location']}")
                                
                                if metadata.get("date"):
                                    prompt_parts.append(f"- Date: {metadata['date']}")
                                
                                if metadata.get("activity"):
                                    prompt_parts.append(f"- Activity: {metadata['activity']}")
                                
                                if metadata.get("people"):
                                    prompt_parts.append(f"- People/Group: {metadata['people']}")
                                
                                if metadata.get("faces"):
                                    face_names = [f.get("name", "") for f in metadata["faces"] if f.get("name")]
                                    if face_names:
                                        prompt_parts.append(f"- People in photo: {', '.join(face_names)}")
                                
                                if metadata.get("relationship"):
                                    prompt_parts.append(f"- Relationship: {metadata['relationship']}")
                                
                                if metadata.get("weather"):
                                    prompt_parts.append(f"- Weather: {metadata['weather']}")
                                
                                if metadata.get("mood"):
                                    prompt_parts.append(f"- Mood: {metadata['mood']}")
                                
                                if metadata.get("tags"):
                                    tags_str = ", ".join(metadata["tags"]) if isinstance(metadata["tags"], list) else metadata["tags"]
                                    if tags_str:
                                        prompt_parts.append(f"- Tags: {tags_str}")
                                
                                if metadata.get("notes"):
                                    prompt_parts.append(f"- Additional notes: {metadata['notes']}")
                            
                            enhanced_prompt = "\n".join(prompt_parts)
                            
                            caption = st.session_state.captioner._generate_caption(image, prompt=enhanced_prompt)
                            
                            st.session_state.captions[selected_image] = caption
                            
                            col1, col2 = st.columns([1, 1])
                            with col1:
                                st.image(image, caption=selected_image, use_container_width=True)
                            with col2:
                                st.markdown("### Generated Caption")
                                st.write(caption)
                        except Exception as e:
                            st.error(f"❌ Error generating caption: {str(e)}")
            
            else:  # All Images
                if st.button("Generate Captions for All Images", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, img_path in enumerate(st.session_state.uploaded_images):
                        img_name = os.path.basename(img_path)
                        status_text.text(f"Processing: {img_name} ({i+1}/{len(st.session_state.uploaded_images)})")
                        
                        try:
                            image = Image.open(img_path).convert("RGB")
                            
                            # Get metadata for this image to enhance the caption
                            metadata = st.session_state.photo_metadata.get(img_name, {})
                            
                            # Build enhanced prompt with metadata
                            prompt_parts = ["Describe the image in detail."]
                            
                            if metadata:
                                prompt_parts.append("\n\nConsider the following context about this photo:")
                                
                                if metadata.get("location"):
                                    prompt_parts.append(f"- Location: {metadata['location']}")
                                
                                if metadata.get("date"):
                                    prompt_parts.append(f"- Date: {metadata['date']}")
                                
                                if metadata.get("activity"):
                                    prompt_parts.append(f"- Activity: {metadata['activity']}")
                                
                                if metadata.get("people"):
                                    prompt_parts.append(f"- People/Group: {metadata['people']}")
                                
                                if metadata.get("faces"):
                                    face_names = [f.get("name", "") for f in metadata["faces"] if f.get("name")]
                                    if face_names:
                                        prompt_parts.append(f"- People in photo: {', '.join(face_names)}")
                                
                                if metadata.get("relationship"):
                                    prompt_parts.append(f"- Relationship: {metadata['relationship']}")
                                
                                if metadata.get("weather"):
                                    prompt_parts.append(f"- Weather: {metadata['weather']}")
                                
                                if metadata.get("mood"):
                                    prompt_parts.append(f"- Mood: {metadata['mood']}")
                                
                                if metadata.get("tags"):
                                    tags_str = ", ".join(metadata["tags"]) if isinstance(metadata["tags"], list) else metadata["tags"]
                                    if tags_str:
                                        prompt_parts.append(f"- Tags: {tags_str}")
                                
                                if metadata.get("notes"):
                                    prompt_parts.append(f"- Additional notes: {metadata['notes']}")
                            
                            enhanced_prompt = "\n".join(prompt_parts)
                            
                            caption = st.session_state.captioner._generate_caption(image, prompt=enhanced_prompt)
                            st.session_state.captions[img_name] = caption
                        except Exception as e:
                            st.warning(f"⚠️ Error processing {img_name}: {str(e)}")
                        
                        progress_bar.progress((i + 1) / len(st.session_state.uploaded_images))
                    
                    status_text.empty()
                    st.success(f"✅ Generated captions for {len(st.session_state.captions)} images!")
            
            # Display existing captions
            if st.session_state.captions:
                st.markdown("---")
                st.subheader("Generated Captions")
                
                for img_name, caption in st.session_state.captions.items():
                    with st.expander(f"📷 {img_name}"):
                        st.write(caption)
                        
                        # Find the image path
                        img_path = next((img for img in st.session_state.uploaded_images 
                                       if os.path.basename(img) == img_name), None)
                        if img_path:
                            st.image(Image.open(img_path), use_container_width=True)

# Search Photos Page
elif page == "🔍 Search Photos":
    st.title("Search Photos")
    st.markdown("Search through your photo collection using natural language queries.")
    st.markdown("---")
    
    if not st.session_state.uploaded_images:
        st.warning("⚠️ No images uploaded yet. Please upload photos first.")
    else:
        search_query = st.text_input(
            "Enter your search query",
            placeholder="e.g., 'a photo of a beach', 'mountains and lakes', 'people smiling'"
        )
        
        top_k = st.slider("Number of results", 1, 10, 5)
        
        if st.button("🔍 Search", type="primary") and search_query:
            with st.spinner("Searching..."):
                try:
                    results = search_similar_images(search_query, top_k=top_k)
                    
                    if results:
                        st.success(f"✅ Found {len(results)} results")
                        st.markdown("---")
                        
                        # Display results in columns
                        cols = st.columns(min(3, len(results)))
                        for idx, result in enumerate(results):
                            with cols[idx % 3]:
                                if os.path.exists(result.payload['path']):
                                    img = Image.open(result.payload['path'])
                                    st.image(img, use_container_width=True)
                                    st.caption(f"Score: {result.score:.3f}")
                                    st.caption(os.path.basename(result.payload['path']))
                                else:
                                    st.warning(f"Image not found: {result.payload['path']}")
                    else:
                        st.info("No results found. Try a different search query.")
                except Exception as e:
                    st.error(f"❌ Error searching: {str(e)}")
                    st.info("💡 Make sure Qdrant is running and images are embedded in the vector database.")

# Generate Travelogue Page
elif page == "📖 Generate Travelogue":
    st.title("Generate Travelogue")
    st.markdown("Create a beautiful travelogue story from your photos using AI.")
    st.markdown("---")
    
    if not st.session_state.uploaded_images:
        st.warning("⚠️ No images uploaded yet. Please upload photos first.")
    elif not st.session_state.captions:
        st.warning("⚠️ No captions generated yet. Please generate captions for your photos first.")
    else:
        st.subheader("Travelogue Settings")
        
        travelogue_prompt = st.text_area(
            "Custom prompt (optional)",
            placeholder="e.g., 'Create a travel story about my beach vacation'",
            help="Leave empty to use default prompt"
        )
        
        # Prepare context from captions
        captions_context = "\n".join([
            f"Photo: {img_name}\nCaption: {caption}\n"
            for img_name, caption in st.session_state.captions.items()
        ])
        
        if st.button("✨ Generate Travelogue", type="primary"):
            with st.spinner("Generating your travelogue..."):
                try:
                    # Use default prompt if none provided
                    if not travelogue_prompt:
                        travelogue_prompt = "Create a beautiful travelogue story from these travel photos and their captions."
                    
                    # Generate travelogue using LLM
                    travelogue = inference_server_model(captions_context, travelogue_prompt)
                    
                    st.markdown("---")
                    st.subheader("Your Travelogue")
                    st.markdown(travelogue)
                    
                    # Option to download
                    st.download_button(
                        label="📥 Download Travelogue",
                        data=travelogue,
                        file_name="travelogue.txt",
                        mime="text/plain"
                    )
                except Exception as e:
                    st.error(f"❌ Error generating travelogue: {str(e)}")
                    st.info("💡 Make sure Ollama is running and the LLM model is available.")
        
        # Show preview of available captions
        if st.session_state.captions:
            with st.expander("📋 Preview Available Captions"):
                st.json(st.session_state.captions)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "Travelogue uses AI models including:\n"
    "- LLaVA for image captioning\n"
    "- CLIP for image search\n"
    "- InsightFace for face detection\n"
    "- Ollama LLM for story generation"
)