import cv2
from ultralytics import YOLO
import time

def calculateScore(hand):
    score = 0
    numAces = 0
    aces = ["Ac", "Ad", "Ah", "As"]
    for card in hand:
        if card in aces:
            numAces += 1

        score += cards[card]

    # if aces can fit as 11, add the 10 to make it so
    for i in range(numAces):
        if score + 10 <= 21:
            score += 10

    return score


cards = {"10c" : 10, "10d" : 10, "10h" : 10, "10s" : 10,
         "2c" : 2, "2d" : 2, "2h" : 2, "2s" : 2,
         "3c" : 3, "3d" : 3, "3h" : 3, "3s" : 3,
         "4c" : 4, "4d" : 4, "4h" : 4, "4s" : 4,
         "5c" : 5, "5d" : 5, "5h" : 5, "5s" : 5,
         "6c" : 6, "6d" : 6, "6h" : 6, "6s" : 6,
         "7c" : 7, "7d" : 7, "7h" : 7, "7s" : 7,
         "8c" : 8, "8d" : 8, "8h" : 8, "8s" : 8,
         "9c" : 9, "9d" : 9, "9h" : 9, "9s" : 9,
         "Ac" : 1, "Ad" : 1, "Ah" : 1, "As" : 1,
         "Jc" : 10, "Jd" : 10, "Jh" : 10, "Js" : 10,
         "Kc" : 10, "Kd" : 10, "Kh" : 10, "Ks" : 10,
         "Qc" : 10, "Qd" : 10, "Qh" : 10, "Qs" : 10}

# trained model
model = YOLO("best.pt")

cam = cv2.VideoCapture(1)

player = []
dealer = []

first = {}
# first segment, initialize dealer cards
# while True:
#     ret, frame = cam.read()
#     if not ret:
#         break
#
#     frame = cv2.flip(frame, 1)
#
#     # prediction
#     results = model(frame)
#
#     if results:
#         for result in results:
#             numDetections = result.boxes.shape[0]
#
#             for i in range(numDetections):
#                 cls = int(result.boxes.cls[i].item())
#                 name = result.names[cls]
#                 if name not in first:
#                     first[name] = 0
#                     first[name] += 1
#                 else:
#                     first[name] += 1
#
#     for card in first:
#         if first[card] >= 70 and card not in dealer and len(dealer) < 2:
#             dealer.append(card)
#
#     if len(dealer) == 2:
#         break
#
#     # display prediction
#     annotated_frame = results[0].plot()
#
#     cv2.imshow("dealer's cards", annotated_frame)
#
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break
#
# cam.release()
# cv2.destroyAllWindows()
#
# time.sleep(2)
#
# first = {}
# cam = cv2.VideoCapture(1)
# # second segment, initialize player cards
# while True:
#     ret, frame = cam.read()
#     if not ret:
#         break
#
#     frame = cv2.flip(frame, 1)
#
#     # prediction
#     results = model(frame)
#
#     if results:
#         for result in results:
#             numDetections = result.boxes.shape[0]
#
#             for i in range(numDetections):
#                 cls = int(result.boxes.cls[i].item())
#                 name = result.names[cls]
#                 if name not in first:
#                     first[name] = 0
#                     first[name] += 1
#                 else:
#                     first[name] += 1
#
#     for card in first:
#         if first[card] >= 70 and card not in player and len(player) < 2:
#             player.append(card)
#
#     if len(player) == 2:
#         break
#
#     # display prediction
#     annotated_frame = results[0].plot()
#
#     cv2.imshow("player's cards", annotated_frame)
#
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break
#
# cam.release()
# cv2.destroyAllWindows()

playerInit = {}
dealerInit = {}
while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # mirror for natural view

    # split frame into left (player) and right (dealer)
    h, w = frame.shape[:2]
    mid = w // 2
    player_frame = frame[:, :mid]
    dealer_frame = frame[:, mid:]

    # run YOLO detection on both halves
    results_player = model(player_frame)
    results_dealer = model(dealer_frame)

    # read player's initial cards
    if results_player and len(player) < 2:
        for result in results_player:
                numDetections = result.boxes.shape[0]

                for i in range(numDetections):
                    cls = int(result.boxes.cls[i].item())
                    name = result.names[cls]
                    if name not in playerInit:
                        playerInit[name] = 0
                        playerInit[name] += 1
                    else:
                        playerInit[name] += 1

    # add cards to player's hand if they have been on the screen for a certain amount of time
    for card in playerInit:
        if playerInit[card] >= 30 and card not in player and len(player) < 2:
            player.append(card)

    # read dealer's initial cards
    if results_dealer and len(dealer) < 2:
        for result in results_dealer:
                numDetections = result.boxes.shape[0]

                for i in range(numDetections):
                    cls = int(result.boxes.cls[i].item())
                    name = result.names[cls]
                    if name not in dealerInit:
                        dealerInit[name] = 0
                        dealerInit[name] += 1
                    else:
                        dealerInit[name] += 1

    # add cards to dealer's hand if they have been on the screen for a certain amount of time
    for card in dealerInit:
        if dealerInit[card] >= 30 and card not in dealer and len(dealer) < 2:
            dealer.append(card)

    if len(player) == 2 and len(dealer) == 2:
        break

    # annotate both halves
    annotated_player = results_player[0].plot()
    annotated_dealer = results_dealer[0].plot()

    # merge annotated halves back into one frame
    combined = cv2.hconcat([annotated_player, annotated_dealer])

    # draw line in between and put text
    cv2.line(combined, (mid, 0), (mid, h), (0, 0, 0), 2)
    cv2.putText(combined, "player", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(combined, "score: " + str(calculateScore(player)), (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(combined, "dealer", (mid + 50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.putText(combined, "score: " + str(calculateScore(dealer)), (mid + 50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("Blackjack - Player (left) | Dealer (right)", combined)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()

print("player: ", player)
print("dealer: ", dealer)

playerScore = calculateScore(player)
dealerScore = calculateScore(dealer)

print("player initial score: ", playerScore)
print("dealer initial score: ", dealerScore)


