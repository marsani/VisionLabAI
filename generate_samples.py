import cv2
import numpy as np
import os

os.makedirs('static/samples', exist_ok=True)

def create_road_lane():
    # 600x400 road image
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    # Sky
    for y in range(160):
        color = (int(180 - y*0.5), int(140 - y*0.4), int(100 - y*0.3))
        img[y, :] = color
    # Mountains / Horizon
    cv2.polylines(img, [np.array([[0, 160], [150, 130], [300, 150], [450, 120], [600, 160]])], False, (70, 70, 70), 2)
    cv2.fillPoly(img, [np.array([[0, 160], [150, 130], [300, 150], [450, 120], [600, 160], [600, 180], [0, 180]])], (60, 60, 60))
    # Road surface
    for y in range(160, 400):
        shade = int(45 + (y - 160)*0.15)
        img[y, :] = (shade, shade, shade)
    # Grass roadside
    pts_left_grass = np.array([[0, 160], [220, 160], [0, 400]])
    pts_right_grass = np.array([[600, 160], [380, 160], [600, 400]])
    cv2.fillPoly(img, [pts_left_grass], (34, 110, 34))
    cv2.fillPoly(img, [pts_right_grass], (34, 110, 34))
    
    # Left Solid Yellow Line
    cv2.line(img, (235, 160), (90, 400), (0, 215, 255), 6, cv2.LINE_AA)
    # Right Solid White Line
    cv2.line(img, (365, 160), (510, 400), (240, 240, 240), 6, cv2.LINE_AA)
    # Center Dashed White Line
    for i in range(7):
        t1 = i / 7.0
        t2 = (i + 0.5) / 7.0
        y1 = int(160 + t1 * 240)
        y2 = int(160 + t2 * 240)
        x1 = int(300 + (t1**1.1) * 0)
        x2 = int(300 + (t2**1.1) * 0)
        thick = max(2, int(2 + t1 * 6))
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), thick, cv2.LINE_AA)
        
    cv2.imwrite('static/samples/road_lane.jpg', img)

def create_document():
    # Angled document on dark wooden desk
    img = np.zeros((450, 600, 3), dtype=np.uint8)
    img[:] = (35, 45, 60) # Desk background
    # Add wood grain noise
    noise = np.random.normal(0, 5, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Document coordinates (trapezoid / tilted)
    src_pts = np.array([[120, 80], [490, 60], [530, 390], [80, 370]], dtype=np.float32)
    doc_w, doc_h = 350, 460
    doc_canvas = np.ones((doc_h, doc_w, 3), dtype=np.uint8) * 248
    
    # Draw header and text lines on the doc canvas
    cv2.rectangle(doc_canvas, (30, 30), (320, 70), (40, 70, 180), -1)
    cv2.putText(doc_canvas, "INVOICE #2026-CV04", (45, 58), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Text lines
    for line_y in range(110, 420, 25):
        w = np.random.randint(180, 280)
        cv2.line(doc_canvas, (35, line_y), (35 + w, line_y), (80, 80, 80), 3, cv2.LINE_AA)
    
    # Barcode
    for bx in range(40, 260, 6):
        bw = np.random.choice([2, 3, 4])
        cv2.rectangle(doc_canvas, (bx, 380), (bx + bw, 430), (20, 20, 20), -1)

    # Warp into perspective on desk
    dst_pts = src_pts
    orig_pts = np.array([[0, 0], [doc_w, 0], [doc_w, doc_h], [0, doc_h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(orig_pts, dst_pts)
    warped_doc = cv2.warpPerspective(doc_canvas, M, (600, 450))
    
    # Mask
    mask = np.zeros((450, 600), dtype=np.uint8)
    cv2.fillPoly(mask, [src_pts.astype(np.int32)], 255)
    
    img[mask > 0] = warped_doc[mask > 0]
    # Draw shadow
    cv2.imwrite('static/samples/document.jpg', img)

def create_license_plate():
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    img[:] = (50, 50, 55) # Car body
    
    # Car bumper curves
    cv2.rectangle(img, (50, 80), (550, 340), (70, 75, 80), -1)
    cv2.rectangle(img, (80, 110), (520, 310), (40, 42, 45), -1)
    
    # License plate container
    cv2.rectangle(img, (130, 160), (470, 270), (25, 25, 25), -1)
    cv2.rectangle(img, (140, 170), (460, 260), (245, 245, 245), -1) # White Plate
    cv2.rectangle(img, (143, 173), (457, 257), (20, 20, 20), 2)
    
    # Text on plate
    cv2.putText(img, "B  1234  CVN", (160, 225), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (15, 15, 15), 3, cv2.LINE_AA)
    cv2.putText(img, "08 . 29", (270, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
    
    # Add headlights / reflections
    cv2.ellipse(img, (80, 200), (30, 70), 0, 0, 360, (200, 80, 60), -1)
    cv2.ellipse(img, (520, 200), (30, 70), 0, 0, 360, (200, 80, 60), -1)
    
    cv2.imwrite('static/samples/license_plate.jpg', img)

def create_xray():
    # Simulated Hand X-Ray with bone contours and soft tissue
    img = np.zeros((500, 400), dtype=np.uint8)
    # Background slight glow
    cv2.circle(img, (200, 250), 220, 30, -1)
    
    # Soft tissue contour
    pts_tissue = np.array([
        [150, 490], [140, 320], [100, 250], [90, 180], [115, 150], [130, 220],
        [145, 130], [165, 80], [185, 80], [195, 170],
        [210, 60], [235, 60], [240, 170],
        [260, 100], [280, 100], [285, 200],
        [305, 160], [325, 170], [315, 260],
        [290, 340], [280, 490]
    ])
    cv2.fillPoly(img, [pts_tissue], 75)
    img = cv2.GaussianBlur(img, (25, 25), 0)
    
    # Bone structures (high intensity)
    bone_canvas = np.zeros((500, 400), dtype=np.uint8)
    
    # Palm bones (Metacarpals)
    for bx in [150, 180, 215, 250, 280]:
        cv2.line(bone_canvas, (bx, 280), (bx + int((bx-215)*0.2), 370), 180, 12)
        
    # Finger bones (Phalanges)
    # Thumb
    cv2.line(bone_canvas, (135, 260), (110, 200), 190, 10)
    cv2.line(bone_canvas, (110, 200), (105, 165), 190, 8)
    # Index
    cv2.line(bone_canvas, (175, 260), (170, 190), 200, 10)
    cv2.line(bone_canvas, (170, 185), (168, 135), 200, 9)
    cv2.line(bone_canvas, (168, 130), (166, 95), 200, 7)
    # Middle
    cv2.line(bone_canvas, (215, 260), (215, 180), 210, 11)
    cv2.line(bone_canvas, (215, 175), (215, 120), 210, 10)
    cv2.line(bone_canvas, (215, 115), (215, 75), 210, 8)
    # Ring
    cv2.line(bone_canvas, (255, 260), (258, 190), 200, 10)
    cv2.line(bone_canvas, (258, 185), (260, 140), 200, 9)
    cv2.line(bone_canvas, (260, 135), (262, 110), 200, 7)
    # Little
    cv2.line(bone_canvas, (285, 275), (295, 220), 180, 8)
    cv2.line(bone_canvas, (295, 215), (305, 175), 180, 7)
    
    # Wrist bones
    cv2.ellipse(bone_canvas, (210, 420), (50, 30), 0, 0, 360, 190, -1)
    
    bone_canvas = cv2.GaussianBlur(bone_canvas, (7, 7), 0)
    img = cv2.add(img, bone_canvas)
    
    # Convert to 3-channel
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.imwrite('static/samples/xray_bone.jpg', img_bgr)

def create_coins():
    img = np.zeros((400, 500, 3), dtype=np.uint8)
    img[:] = (40, 45, 50)
    # Add some background texture
    grid = np.zeros((400, 500), dtype=np.uint8)
    for x in range(0, 500, 30):
        cv2.line(img, (x, 0), (x, 400), (48, 53, 58), 1)
    for y in range(0, 400, 30):
        cv2.line(img, (0, y), (500, y), (48, 53, 58), 1)

    coins = [
        ((120, 120), 55, (40, 180, 220), "500"),
        ((250, 100), 45, (160, 160, 170), "200"),
        ((380, 140), 65, (50, 190, 230), "1000"),
        ((180, 260), 60, (50, 190, 230), "1000"),
        ((320, 270), 50, (170, 170, 180), "500"),
        ((410, 320), 40, (150, 150, 160), "100"),
        ((90, 320), 45, (160, 160, 170), "200"),
    ]
    
    for center, radius, color, text in coins:
        # Shadow
        cv2.circle(img, (center[0] + 5, center[1] + 5), radius, (20, 22, 25), -1)
        # Coin body
        cv2.circle(img, center, radius, color, -1)
        # Inner rim
        cv2.circle(img, center, radius - 6, (int(color[0]*0.85), int(color[1]*0.85), int(color[2]*0.85)), 2)
        # Text
        cv2.putText(img, text, (center[0] - 22, center[1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2, cv2.LINE_AA)
        
    cv2.imwrite('static/samples/coins.jpg', img)

def create_circuit_board():
    img = np.zeros((400, 500, 3), dtype=np.uint8)
    img[:] = (20, 80, 35) # PCB Green
    
    # Chip IC
    cv2.rectangle(img, (180, 140), (320, 260), (30, 30, 30), -1)
    cv2.putText(img, "ARM Cortex-M4", (190, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(img, "STM32F401", (210, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA)
    
    # IC Pins
    for y in range(155, 250, 15):
        # Left pins
        cv2.rectangle(img, (160, y), (180, y+6), (190, 195, 205), -1)
        cv2.line(img, (160, y+3), (60, y+3 + np.random.randint(-20, 20)), (40, 170, 210), 2)
        # Right pins
        cv2.rectangle(img, (320, y), (340, y+6), (190, 195, 205), -1)
        cv2.line(img, (340, y+3), (440, y+3 + np.random.randint(-20, 20)), (40, 170, 210), 2)

    # Solder pads / vias
    for _ in range(40):
        vx = np.random.randint(30, 470)
        vy = np.random.randint(30, 370)
        if not (170 < vx < 330 and 130 < vy < 270):
            cv2.circle(img, (vx, vy), 6, (40, 180, 220), -1)
            cv2.circle(img, (vx, vy), 2, (15, 50, 20), -1)

    cv2.imwrite('static/samples/circuit_board.jpg', img)

create_road_lane()
create_document()
create_license_plate()
create_xray()
create_coins()
create_circuit_board()
print("Sample images generated successfully!")
