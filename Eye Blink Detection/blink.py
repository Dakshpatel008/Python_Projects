import cv2

# Load the pre-trained eye cascade classifier
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Variables to track blink status
blink_counter = 0
is_blinking = False

# Define a function to detect eye blinks
def detect_blinks(frame):
    global blink_counter, is_blinking

    # Convert the frame to grayscale for better detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect eyes in the frame
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Loop over the detected eyes
    for (x, y, w, h) in eyes:
        # Draw rectangles around the eyes for visualization
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Check if there are no eyes detected
    if len(eyes) == 0:
        is_blinking = False

    # Check if the previous frame had eyes but the current frame doesn't
    elif not is_blinking and len(eyes) == 1:
        is_blinking = True
        blink_counter += 1

    # Display the blink counter on the frame
    cv2.putText(frame, "Blinks: {}".format(blink_counter), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the frame with detected eyes
    cv2.imshow("Eye Blink Detection", frame)

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame from the webcam
    ret, frame = cap.read()

    # Break the loop if no frame is captured
    if not ret:
        break

    # Detect eye blinks in the current frame
    detect_blinks(frame)

    # Check for the 'q' key to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
