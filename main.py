import cv2
import numpy as np

from candidates_handler import CandidatesHandler
from img_processing import mog_mask_processing, pre_process, diff_process, check_illumination_changes, cso_img_process
from states import MaskState, CSOState
kernel_2_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
kernel_5_5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
short_counter = 0
long_counter = 0
first_short = False
first_long = False
pso_lock = False
long = cv2.createBackgroundSubtractorMOG2()
short = cv2.createBackgroundSubtractorMOG2()
learning_rate_long = 0.00005
learning_rate_short = 0.008
cap = cv2.VideoCapture('video5.avi')
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)
cap.set(1, 200)
ret, img = cap.read()
averageValue1 = np.float32(img)
final_csos = np.zeros((img.shape[0], img.shape[1]), np.uint8)
candidates_handler = CandidatesHandler()

first_pso = True
pso_frame_num = 0
pso_counter = 0

first_cso = False
cso_frame_num = 0
cso_counter = 0

while (1):
    # read frames
    rgb = cap.read()[1]
    img = pre_process(rgb)
    video_frame_processed = cso_img_process(rgb)
    cv2.imshow('video_frame_processed',video_frame_processed)
    cv2.accumulateWeighted(rgb, averageValue1, 0.001)
    lb = cv2.convertScaleAbs(averageValue1)

    long_mask = long.apply(img, None, learning_rate_long)
    long_mask = mog_mask_processing(long_mask, kernel=kernel_2_2)

    short_mask = short.apply(img, None, learning_rate_short)
    short_mask = mog_mask_processing(short_mask, kernel=kernel_2_2)

    diff = cv2.bitwise_xor(src1=short_mask, src2=long_mask)
    diff = diff_process(diff)

    if check_illumination_changes(diff):
        learning_rate_long = 1
        learning_rate_short = 1
    elif not check_illumination_changes(diff):
        learning_rate_long = 0.00005
        learning_rate_short = 0.008

    cv2.imshow('diff', diff)
    cv2.imshow('long_mask', long_mask)
    cv2.imshow('short_mask', short_mask)

    " --- --- --- PSO to CSO --- --- --- "

    if not pso_lock:
        if first_pso:
            state = candidates_handler.extract_candidates(diff_mask=diff, rgb_frame=rgb, prev_frames=lb, initital=True)
            if state == MaskState.NOT_VALID_MASK:
                continue
            else:
                first_pso = False

        else:
            pso_counter = pso_counter + 1
            if pso_counter == 10:
                state = candidates_handler.extract_candidates(diff_mask=diff, rgb_frame=rgb)
                if state == MaskState.VALID_MASK:
                    pso_frame_num = pso_frame_num + 1
                else:
                    pso_frame_num = 0
                    first_pso = True
                pso_counter = 0

        if pso_frame_num == 5:
            print('pso done')
            csos_mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
            csos = candidates_handler.find_csos()
            if len(csos) != 0:
                for cso in csos:
                    cv2.drawContours(csos_mask, [cso], 0, 255, -1)
            csos_filtered = candidates_handler.csos_filter(csos_mask, rgb)
            final_csos = csos_filtered
            first_pso = False
            pso_lock = True
            pso_frame_num = 0
            first_cso = True
            cv2.imshow('csos_mask', csos_mask)
            cv2.imshow('csos_filtered', csos_filtered)

    " --- --- --- CSO to SO --- --- --- "
    if first_cso:
        cso_counter = cso_counter + 1
        if cso_counter == 20:
            cso_frame_num = cso_frame_num + 1
            video_frame_processed = cso_img_process(rgb)
            state = candidates_handler.cso_to_so(final_csos, rgb, diff ,cso_frame_num,
                                                 initial=True if cso_frame_num == 1 else False)
            print('state: {}, cso_frame_num: {}'.format(state,cso_frame_num))
            cv2.imshow('final_csos',final_csos)
            if state == CSOState.BREAK or state == CSOState.CSOS_NOT_VALID:
                cso_counter = 0
                cso_frame_num = 0
                first_cso = False
                first_pso = True
                pso_lock = False
            else:
                cso_counter = 0
                if isinstance(state, list):
                    print('sos done')
                    sos = state
                    cso_frame_num = 0
                    first_cso = False
                    first_pso = True
                    pso_lock = False
                    count = 0
                    for sovf, csodf in sos:
                        count = count + 1
                        cv2.imshow('sovf : {}'.format(count), sovf)
                        cv2.imshow('csodf : {}'.format(count), csodf)
                    cv2.waitKey(0)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
