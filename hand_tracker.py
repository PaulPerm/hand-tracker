import time #for timestamps and measuring fps
import cv2 # handle camera, drawing and the window
import mediapipe as mp # main package, for image and image format
from mediapipe.tasks.python import BaseOptions, vision # holds teh settings page every media pipe task shares 



MODEL = "hand_landmarker.task"
TIPS = [4, 8, 12, 16, 20] #thumb, index, middle, ring, pinky tips of fingers
PIPS = [3, 6, 10, 14, 18] #the join just below each tip
CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS # helps with connect lines to draw skeleton.


#pythagreon to show straight line between two landmarks
def dist(a, b):
    return ((a.x - b.x) ** 2 + (a.y -b.y) **2) ** 0.5


#lm is list of the 21 landmakrs for one hand works left to right always
def fingers_up(lm):
    # thumb bends sidewats so compare distanceto the pinky knuckle instead of height
    up = [dist(lm[4], lm[17]) > dist(lm[3], lm[17])]

    #other fingers: tip above its middle joint = up (y grows downwatds in images)
    for tip, pip in zip(TIPS[1:], PIPS[1:]):
        up.append(lm[tip].y < lm[pip].y)
    return up

def pinching(lm):
    threshold = 0.05
    return dist(lm[4],lm[8]) < threshold
#look up table
"""
has to be tuple and not a list because dictonary keys need to be immutable

"""
def name_gesture(up):
    patterns = {
        (0, 0, 0, 0, 0): "Fist",
        (1, 1, 1, 1, 1): "Open Palm",
        (0, 1, 0, 0, 0): "Pointing",
        (0, 1, 1, 0, 0): "Peace",
        (1, 0, 0, 0, 0): "Thumbs up",
        (0, 1, 0, 0, 1): "Rock on",
        (1, 0, 0, 0, 1): "Call me",
    }   
    return patterns.get(tuple(int(u) for u in up), f"{sum(up)} fingers")



def draw_hand(frame, lm):
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in lm] #builds list converts every landmark into coordinates
    for c in CONNECTIONS: #draws line between each point
        cv2.line(frame, pts[c.start], pts[c.end], (255, 255, 255), 2)# image to draw on, start point, end point, color of line, and thickenss 

    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (0, 200, 255), -1)
    return pts




def main():
    #settings object for the dectecetor
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path = MODEL),
        running_mode=vision.RunningMode.VIDEO, # mde to get frames in order tracks teh hand between frames. which is faster and smoothers needs timestamps
        num_hands=2,
    )
    cap = cv2.VideoCapture(0)
    prev = time.time()

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #makes the skin less blue to increase accuracy. 
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb) #wraps teh raw arrya in Image types and tells it to format is standard RGB 
            results = landmarker.detect_for_video(mp_image, int(time.monotonic() * 1000)) # runs the model in VIDEO mode because it needs a timestamp 

            for lm in results.hand_landmarks: #runs once per detected hand if there is no had the list is empty and the loop just doesnt run 
                pts = draw_hand(frame, lm) 
                label = name_gesture(fingers_up(lm))
                if pinching(lm):
                    label = "Pinching"
                wx, wy = pts[0]
                cv2.putText(frame, label, (wx -50, wy + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
            now = time.time()
            fps = 1 / (now -prev)
            prev = now
            cv2.putText(frame, f"FPS: {fps:.0f}", (10,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.imshow("Hand Tracker (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()