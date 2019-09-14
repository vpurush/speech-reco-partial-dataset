import difflib



def findNearestWord(charSequence, cutoffValue):
    # print("cutoffValue ", cutoffValue)
    return difflib.get_close_matches(charSequence, [
        "aim",
        "air",
        "along",
        "anger",
        "answer",
        "badge",
        "ban",
        "bank",
        "belong",
        "cage",
        "caring",
        "change",
        "chair",
        "chain",
        "clan",
        "cleaning",
        "coming",
        "cutting",
        "edge",
        "farming",
        "forge",
        "gain",
        "gaming",
        "grading",
        "hack",
        "hand",
        "hard",
        "hat",
        "have",
        "hue",
        "huge",
        "invading",
        "jail",
        "joking",
        "large",
        "long",
        "man",
        "racing",
        "raid",
        "rail",
        "rain",
        "raise",
        "range",
        "stain",
        "strange",
        "strength",
        "thing",
        "vain",
        "vase",
        "young"
    ], n = 1, cutoff=cutoffValue)

def findNearestWordList(charSequenceList):
    nearestWordList = []

    # print("\n findNearestWordList")
    cutoffValue = 0.95
    while True:
        # print("cutoffVale", cutoffValue)
        for charSequence in list(charSequenceList):
            # print("inside for")
            nearestWordForSeq = findNearestWord(charSequence, cutoffValue)
            # print("findNearestWord", charSequence, nearestWordForSeq)

            if nearestWordForSeq:
                for nearestWord in nearestWordForSeq:
                    if nearestWord not in nearestWordList:
                        # print(nearestWord)
                        nearestWordList.append(nearestWord)

            if len(nearestWordList) >= 5:
                break

        
        if len(nearestWordList) >= 5:
            break
        elif cutoffValue > 0.1:
            cutoffValue = cutoffValue - 0.05
        else:
            break

    print("nearestWordList", nearestWordList)