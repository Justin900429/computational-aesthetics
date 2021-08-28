from CA import CA

if __name__ == "__main__":
    # Create CA objects
    img_path = "black.jpeg"
    ca = CA(img_path)
    res = ca.compute_ca()
    print(f"Number of features: {len(res)}\nFeatures: {res}")

    # Update the image
    img_path = "coarse.png"
    ca.update(img_path)
    new_res = ca.compute_ca()
    print(f"Number of features: {len(new_res)}\nFeatures: {new_res}")
