# Family Reference Flow (Streamlit) Plan

Goal: new Streamlit flow (`main2.py`) that helps build a family face reference set. User uploads family photos, the app crops each face, collects metadata, stores originals and crops under `/home/anuragd/labshare/travelogue`, generates face embeddings, and writes vectors plus metadata to Qdrant.

## Target storage layout (on disk)
- Base: `/home/anuragd/labshare/travelogue`
- `family_photos/originals/` — saved uploads (keep full images)
- `family_photos/crops/` — face crops (one per detected face)
- `family_photos/metadata/` — JSON per upload/crop and an optional consolidated index
- File names: include timestamp + UUID to avoid collisions; keep the original filename in metadata.

## Qdrant collection
- Collection name: `family_faces`
- Vector size: use the length of `embedding` returned by `get_face_embeddings`; create collection if missing.
- Distance: cosine (typical for face embeddings).
- Payload fields (per face):
  - `person_name`, `relation` (e.g., wife, son, daughter), optional `family_id`
  - `notes` (free text), `source_image_path`, `crop_path`, `bbox`, `det_score`
  - `captured_at` (EXIF/entered date), `uploaded_at` (timestamp), `original_filename`
  - `hash` or `photo_id` to deduplicate per photo

## Streamlit UX (main2.py)
1. **Upload family photo(s)**: accept multiple images; immediately save to `originals/`.
2. **Detect faces**: run InsightFace (`get_face_embeddings`) to get bboxes + embeddings; store in session.
3. **Review & crop**: show each detected face with bounding boxes; generate and display crops for each face.
4. **Collect metadata per face**: fields for name, relation, optional date (EXIF default), notes; allow skipping faces.
5. **Persist**:
   - Save crop files to `crops/`
   - Write per-photo metadata JSON under `metadata/`
6. **Index in Qdrant**:
   - Ensure `family_faces` collection exists (vector size from embedding length)
   - Upsert each face embedding with payload (metadata + file paths + bbox)
7. **Feedback**: success/failure messages, counts of faces indexed, links to saved files.

## Implementation steps
1. Scaffold `main2.py` (keep `main.py` untouched): new page for “Family Reference” with session keys for uploads, detections, crops, metadata.
2. Add disk helpers: ensure target folders exist; hash filenames; crop-and-save utility using PIL with bbox from InsightFace.
3. Build upload + detection flow: save originals, run `get_face_embeddings`, keep results in session, build crops in memory for preview.
4. Metadata form per face: name/relation/date/notes; allow “skip” or “save”.
5. Persist + index handler: on submit, write files/metadata JSON, ensure Qdrant collection, upsert embeddings with payload.
6. (Optional) Add simple viewer for what’s been saved (grid of crops + metadata), and a “clear session state” button that does not delete disk files.

## Notes / assumptions
- InsightFace utils already return `facial_area`, `det_score`, and `embedding`; confirm shape before creating the collection.
- No deletion of disk files per user note; only session resets are allowed.
- Qdrant connectivity/config is assumed to be handled by existing utilities; we will reuse them or add a minimal client if needed.
