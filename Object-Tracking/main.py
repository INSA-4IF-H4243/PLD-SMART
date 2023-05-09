import cv2

import util

VIDEO = "video_input2"
VIDEO_INPUT = f"inputs/{VIDEO}.mp4"
VIDEO_OUTPUT = f"outputs/{VIDEO}.avi"

factor = 0.49
parameters = {
    "filter": {"iterations": 5, "shape": (10, 10)},  # brush size
    "substractor": {"history": 200, "threshold": 200},
}

subtractors = ["GMG", "MOG", "MOG2", "KNN", "CNT"]
subtractor = util.subtractor(subtractors[2], parameters["substractor"])

capture = cv2.VideoCapture(VIDEO_INPUT)
exist, frame = capture.read()

while exist:
    transformations = []

    transformations.append(cv2.resize(frame, (0, 0), fx=factor, fy=factor))
    # cv2.imshow("frame", transformations[-1])

    transformations.append(transformations[-1][45:500, 140:810])
    # cv2.imshow("test", transformations[-1])

    transformations.append(cv2.cvtColor(transformations[-1], cv2.COLOR_BGR2GRAY))
    # cv2.imshow("gray", transformations[-1])

    transformations.append(subtractor.apply(transformations[-1]))
    cv2.imshow("mask", transformations[-1])

    transformations.append(
        util.filter(transformations[-1], "closing", parameters["filter"])
    )
    cv2.imshow("closing", transformations[-1])

    # transformations.append(util.filter(transformations[-1], "dilation", parameters["filter"]))
    # cv2.imshow("dilation", transformations[-1])

    # transformations.append(cv2.medianBlur(transformations[-1], 5))
    # cv2.imshow("blur", transformations[-1])

    contours, hierarchy = cv2.findContours(
        transformations[-1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1:
            x, y, w, h = cv2.boundingRect(contour)
            if area > 20:
                cv2.rectangle(
                    transformations[0], (x + 140 - 10, y + 45 - 10), (x + 140 + w, y + 45 + h), (0, 0, 255), 2
                )  # players
            else:
                cv2.rectangle(
                    transformations[0], (x + 140 - 10, y + 45 - 10), (x + 140 + w, y + 45 + h), (0, 255, 0), 2
                )  # ball

    cv2.rectangle(transformations[0], (140, 45), (810, 500), (255, 0, 0), 2)
    cv2.imshow("frame", transformations[0])

    # codec = cv2.VideoWriter_fourcc(*'XVID')
    # height = transformations[0].shape[0]
    # width = transformations[0].shape[1]
    # writer = cv2.VideoWriter(VIDEO_OUTPUT, codec, 25, (width, height), False)

    exist, frame = capture.read()
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
