import time
import cv2
import numpy as np

# Constants from https://github.com/thsant/3dmcap/blob/master/resources/Logitech-C920.yaml
focal_length = 609.710537
principal_point = (303.200522, 258.905227)
# 315.20752998 255.90879313
# camera_matrix = np.array([
#     [609.710537, 0, principal_point[0]],
#     [0, 606.011374, principal_point[1]],
#     [0, 0, 1]
# ])
camera_matrix = np.array([
    [627.32011186, 0, 315.20752998],
    [0, 626.92426419, 255.90879313],
    [0, 0, 1]
])

dist_coefs = [0.05529612, -0.24132761, 0.00290915, -0.00184282, 0.84908901]

confidence_requirement = 0.9

diag = np.array([
    [-1, 0, 0],
    [0, -1, 0],
    [0, 0, -1]
])

final_coord = []

R_pos = np.zeros(shape=(3, 3))
t_pos = np.zeros(shape=(3, 1))

cam = cv2.VideoCapture(0)

feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

fast = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)

if not cam.isOpened():
    raise ValueError("Error: Could not open camera")
ret, last_frame = cam.read()
previous_gray = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
p0 = fast.detect(previous_gray)
p0 = np.array([x.pt for x in p0], dtype=np.float32).reshape(-1, 1, 2)
i = 0
while i != 1:
    ret, frame = cam.read()
    if not ret:
        break
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    if len(p0) <= 2000:
        p0 = fast.detect(previous_gray)
        p0 = np.array([x.pt for x in p0], dtype=np.float32).reshape(-1, 1, 2)

    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, p0, None, winSize=(21, 21),
                                           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.01))

    if p1 is not None:
        features_new = p1[st == 1]
        features_old = p0[st == 1]

        for pt1, pt2 in zip(features_old, features_new):
            frame = cv2.circle(frame, (int(pt1[0]), int(pt1[1])), 1, color=(0, 255, 0), thickness=1)
            frame = cv2.circle(frame, (int(pt2[0]), int(pt2[1])), 2, color=(0, 0, 255), thickness=1)
        cv2.imwrite('test.png', frame)

        if len(features_old) != 0 and len(features_new) != 0 and len(p0) != 0:
            print("im here")
            E, mask = cv2.findEssentialMat(features_new, features_old, camera_matrix, cv2.RANSAC,
                                           confidence_requirement,
                                           1.0)
            try:
                _, R, t, mask = cv2.recoverPose(E, features_new, features_old, focal=627.32011186, pp=(315.20752998, 255.90879313))
                print(t.T)
                if R_pos.any() == np.zeros(shape=(3, 3)).any():
                    R_pos = R
                else:
                    t_pos += t
                    R_pos = R.dot(R_pos)
                    adj_coord = np.matmul(diag, t_pos)
                    final_coord = adj_coord.flatten()
                    print(final_coord)
            except:
                ...

    previous_gray = current_gray.copy()

    mask = frame.copy()
    if p0 is not None:
        for point in p0:
            point = point[0]
            mask = cv2.circle(mask, (int(point[0]), int(point[1])), 1, color=(0, 255, 0), thickness=1)
    mask = cv2.putText(mask, str(final_coord), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('frame', mask)
    # i = 1
cam.release()

cv2.destroyAllWindows()
