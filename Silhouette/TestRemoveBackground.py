import ImageTreatment

# Path: Silhouette\TestRemoveBackground.py


def main():
    image = ImageTreatment.load_image("..\\image_input\\tennis1.png")
    #result = ImageTreatment.remove_background(image)
    #result = ImageTreatment.remove_blue_background(image)
    result = ImageTreatment.remove_background_adaptive(image)
    return


if __name__ == "__main__":
    main()
