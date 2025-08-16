# traininglabels -> array
def loadLabelsFile(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    cleaned_lines = [line.strip() for line in lines]
    return cleaned_lines

# trainingimages -> convert char to integers 
def IntegerConversionFunction(character):
    """
    Helper function for file reading.
    """
    if(character == ' '):
        return 0
    elif(character == '+'):
        return 1
    elif(character == '#'):
        return 2 

# trainingimages -> read digits -> array
def loadImagesFile(filename, num_labels, cols):
    with open(filename, 'r') as file:
        lines = file.readlines()  # Preserve all lines, including blanks

    total_lines = len(lines)

    if total_lines % num_labels != 0:
        raise ValueError("Total lines is not divisible by number of labels. Check file integrity.")

    rows_per_image = total_lines // num_labels  # This is your image height (H)
    sections = []

    for i in range(num_labels):
        section = []
        for j in range(rows_per_image):
            line_index = i * rows_per_image + j
            line = lines[line_index].rstrip('\n')
            row = [IntegerConversionFunction(c) for c in line.ljust(cols)[:cols]]
            section.append(row)
        sections.append(section)

    # with open('loaddata.txt', 'w') as f:
    #     for i, section in enumerate(sections):
    #             for row in section:
    #                 f.write(''.join(str(cell) for cell in row) + '\n')
    #             if i != len(sections) - 1:
    #                 f.write('\n')
                      # f.write(''.join('0' for _ in range(cols)) + '\n')  # add a blank row between sections

    #print(len(sections))
    return sections


def _test():
    print("you got this christine! <3")

if __name__ == "__main__":
    digitsLabels = loadLabelsFile("data/digitdata/traininglabels")
    digitImages = loadImagesFile("data/digitdata/trainingimages", len(digitsLabels), 28)
    # digitsLabels = loadLabelsFile("data/digitdata/testlabels")
    # digitImages = loadImagesFile("data/digitdata/testimages", len(digitsLabels), 28)
    # digitsLabels = loadLabelsFile("data/digitdata/validationlabels")
    # digitImages = loadImagesFile("data/digitdata/validationimages", len(digitsLabels), 28)

    # faceLabels = loadLabelsFile("data/facedata/facedatavalidationlabels")
    # faceImages = loadImagesFile("data/facedata/facedatavalidation", len(faceLabels), 60)
    # faceLabels = loadLabelsFile("data/facedata/facedatatestlabels")
    # faceImages = loadImagesFile("data/facedata/facedatatest", len(faceLabels), 60)
    # faceLabels = loadLabelsFile("data/facedata/facedatatrainlabels")
    # faceImages = loadImagesFile("data/facedata/facedatatrain", len(faceLabels), 60)
    
    # print(len(digitsLabels))
    # print(len(faceLabels))
    # print(faceLabels)
    _test()