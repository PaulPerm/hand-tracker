import cv2

cap = cv2.VideoCapture(0) #opens a video source can be a camer or a stored video file or a stream url
while True: #series of still images show quickly about 30 fps this runse once per fram untill somehthign breask
    ok, frame = cap.read() # grab a frame which is a NumPY array 
    if not ok: # handles failed read of a frame 
        print("No camera frame"); break
    frame = cv2.flip(frame,1) #mirrors image 
    cv2.imshow("Camera test (q to quit)", frame) #shows teh frame 
    if cv2.waitKey(1) & 0xFF == ord("q"): #waits for key press, keeps lostes 9 bits of code, 
        break
cap.release()
cv2.destroyAllWindows()