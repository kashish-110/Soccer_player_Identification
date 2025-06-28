# Soccer_player_Identification

## Approach & Methodology

- We use **YOLOv11** (fine-tuned) to detect soccer players in each frame.
- We use **DeepSORT** for tracking players across frames to maintain consistent IDs.
- We adjusted `max_age` and `n_init` to improve re-identification when players leave and re-enter the frame.
- We filtered only relevant classes (`player` class) and applied NMS during detection for cleaner tracking.

## Techniques Tried and Outcomes

 **YOLOv11 + DeepSORT Baseline:**  
Successfully detected and tracked players, assigning unique IDs visible on the output video.

 **Adjusting `max_age` to 300:**  
Allowed tracks to persist longer, reducing unnecessary ID switches.

 **NMS Integration:**  
Reduced duplicate bounding boxes, improving ID consistency.

 **Perfect ID Consistency Not Achieved:**  
Due to occlusions, YOLO missing detections, and re-identification limitations, some ID switches occurred when players left and re-entered the frame.

##  Challenges Encountered

- **Model missing players in some frames** due to occlusion or low visibility.
- **DeepSORT limitations** in re-identification with large pose, scale, and occlusion changes.
- Hard to perfectly align the exact number of players (approx. 22) with consistent IDs throughout without advanced Re-ID models or multi-camera setups.

##  If Incomplete

To fully stabilize IDs:
- Integrate **ByteTrack/StrongSORT** for better ID consistency.
- Train a **Soccer-Specific Re-ID network**.
- Use **jersey number OCR** to anchor IDs to real player identities.
- Use a **multi-camera system** to reduce occlusion issues.

---

##  Conclusion

The pipeline successfully:
- Detects soccer players frame-wise.
- Assigns and maintains player IDs for re-identification in a single feed within assignment constraints.
- Generates a clear, annotated output video for visual validation.

**All code, instructions, and outcomes are self-contained and ready for review.**

