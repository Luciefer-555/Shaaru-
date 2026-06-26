from face_analysis import analyze_face

if __name__ == "__main__":
    profile = analyze_face("SA.jpeg", "test_user")
    print("Skin Tone Label:", profile.skin_tone_label)
    print("Skin Undertone:", profile.skin_undertone)
    print("Hair Color:", profile.hair_color)
    print("Eye Color:", profile.eye_color)
