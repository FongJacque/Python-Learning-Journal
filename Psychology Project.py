import tkinter as tk
from tkinter import simpledialog
from PIL import Image, ImageTk
import random
import openpyxl
from openpyxl import Workbook
import os

# --- Word Pools ---
practice_associated_words = [
    "show", "film", "movie", "stage", "act",
    "play", "scene", "seat", "ticket", "screen",
    "lights", "sound", "watch", "drama", "story",
    "star", "enter", "exit", "crowd", "marquee"
]

practice_unassociated_words = [
    "shirt", "table", "phone", "train", "bread",
    "heart", "anger", "purple", "seven", "sock",
    "climb", "sing", "draw", "ball", "street",
    "child", "friend", "apple", "funny", "brave"
]

# Test 1
associated_words_pool1 = ["roof", "door", "wood", "wall", "home",
                           "yard", "tree", "farm", "grass", "land",
                           "path", "dust", "wind", "lone", "aged",
                           "gray", "chim", "shed", "corn", "gate"]

unassociated_words_pool1 = ["blue", "fish", "jump", "sing", "coin",
                             "silk", "bake", "loud", "five", "star",
                             "milk", "soft", "ring", "fast", "cold",
                             "book", "hand", "town", "shoe", "rain"]

# Test 2
associated_words_pool2 = ["snow", "log", "cabin", "peak", "pine",
                           "cold", "wood", "roof", "hill", "frost",
                           "trail", "winter", "slope", "ridge", "bleak",
                           "stone", "draft", "hush", "stark", "climb",
                           "white", "forest", "frozen", "quiet", "timber",
                           "valley", "powder", "alpine", "rustic", "chill"]

unassociated_words_pool2 = ["chair", "river", "paint", "music", "lemon",
                             "smile", "clock", "paper", "sweet", "light",
                             "dance", "ocean", "plant", "round", "sharp",
                             "voice", "write", "fruit", "shelf", "dream",
                             "plate", "color", "story", "metal", "sugar",
                             "happy", "quick", "floor", "stone", "radio"]

# Test 3
associated_words_pool3 = ["tree", "water", "lake", "shore", "branch",
                           "leaf", "bark", "shade", "green", "river",
                           "boat", "dock", "sun", "light", "grass",
                           "bank", "root", "trunk", "pond", "calm",
                           "still", "ripple", "breeze", "shadow", "nature",
                           "park", "path", "edge", "sky", "cloud",
                           "heron", "willow", "maple", "otter", "lily",
                           "coast", "wave", "pier", "canal", "flow"]

unassociated_words_pool3 = ["table", "phone", "shirt", "train", "bread",
                             "movie", "heart", "anger", "purple", "seven",
                             "sock", "climb", "sing", "draw", "ball",
                             "house", "street", "town", "child", "friend",
                             "apple", "funny", "brave", "sleep", "open",
                             "close", "under", "above", "listen", "speak",
                             "study", "learn", "number", "letter", "begin",
                             "finish", "yellow", "heavy", "empty", "corner"]

# Image paths
practice_image_path = r"C:\Users\fongj\OneDrive\Documents\Python Scripts\Personal Projects\Psychology Project\elijah-mears-CmKxr073k9k-unsplash.jpg"
image_path1 = r"C:\Users\fongj\OneDrive\Documents\Python Scripts\Personal Projects\Psychology Project\monochrome-black-and-white-scenery-landscape-2f21412b6f16c23316ee3fcde5e7000c.jpg"
image_path2 = r"C:\Users\fongj\OneDrive\Documents\Python Scripts\Personal Projects\Psychology Project\gray-scale-shot-wooden-cottage-farm-with-tree-covered-hills.jpg"
image_path3 = r"C:\Users\fongj\OneDrive\Documents\Python Scripts\Personal Projects\Psychology Project\id_Grayscale_vs_Black_White_vs_Monochrome_04.jpg"

# --- UI Setup ---
root = tk.Tk()
root.withdraw()
participant_id = simpledialog.askstring("Participant ID", "Enter Participant ID:")
root.deiconify()
root.title("Psychology Experiment")
root.state('zoomed')

# --- Globals ---
test_order = random.sample([1, 2, 3], 3)
current_test = -1  # Start with practice
response_data = []
current_index = 0
test_words = []
shown_words = []
shown_associated = []
additional_words_for_test = []
in_test_phase = False
current_phase = "instruction"  # Start with instructions
prompt_label = None
instruction_label = None
num_associated_shown_current_test = 0
instruction_label_widget = None

# --- Helper Functions ---
def get_test_data(phase, test_number):
    if phase == "practice":
        return practice_associated_words, practice_unassociated_words, practice_image_path, 20
    elif phase == "test":
        if test_number == 1:
            return associated_words_pool1, unassociated_words_pool1, image_path1, 20
        elif test_number == 2:
            return associated_words_pool2, unassociated_words_pool2, image_path2, 30
        elif test_number == 3:
            return associated_words_pool3, unassociated_words_pool3, image_path3, 40
    return [], [], None, 0

def show_image(image_path):
    img = Image.open(image_path)
    screen_height = root.winfo_screenheight()
    desired_height = screen_height // 2
    aspect_ratio = img.width / img.height
    new_width = int(desired_height * aspect_ratio)
    resized_img = img.resize((new_width, desired_height), Image.LANCZOS)
    return ImageTk.PhotoImage(resized_img)

def start_experiment():
    global current_test, current_index, response_data, current_phase, num_associated_shown_current_test, shown_words, shown_associated

    shown_words = []
    shown_associated = []

    if current_phase == "instruction":
        show_instructions()
    elif current_phase == "practice":
        associated_pool, unassociated_pool, image_path, show_count = get_test_data(current_phase, 0)
        num_associated_shown_current_test = show_count // 2
        start_encoding_phase(image_path, shown_count=show_count, test_number_for_encoding=0)
    elif current_phase == "intermission":
        show_intermission()
    elif current_phase == "test":
        if current_test < 3:
            current_index = 0
            response_data = []
            test_index = test_order[current_test]
            associated_pool, unassociated_pool, image_path, show_count = get_test_data(current_phase, test_index)
            num_associated_shown_current_test = show_count // 2
            start_encoding_phase(image_path, shown_count=show_count, test_number_for_encoding=test_index)
        else:
            reset_ui()
            label = tk.Label(root, text="Thank you! You have completed all tests.", font=("Times New Roman", 28))
            label.pack(expand=True)
            return
    elif current_test >= 3:
        reset_ui()
        label = tk.Label(root, text="Thank you! You have completed all tests.", font=("Times New Roman", 28))
        label.pack(expand=True)
        return

def show_instructions():
    global instruction_label_widget
    reset_ui()
    instruction_text = """You will be shown an image. Underneath the image will be a collection of words.

After 30 seconds, you will be tested on the words shown.


Press right on the keyboard if you saw the word.

Press left on the keyboard if you did not see the word."""
    font = ("Times New Roman", 26)
    instruction_label_widget = tk.Label(root, text=instruction_text, font=font, justify='center')
    instruction_label_widget.pack(expand=True, fill='both')
    root.after(15000, begin_practice)

def show_intermission():
    global instruction_label_widget
    reset_ui()
    intermission_text = """You will now be shown multiple images with varying numbers of words underneath.

    
Press right on the keyboard, if you saw the word.

Otherwise, press left on the keyboard, if you did not see the word."""
    font = ("Times New Roman", 26)
    instruction_label_widget = tk.Label(root, text=intermission_text, font=font, justify='center')
    instruction_label_widget.pack(expand=True, fill='both')
    root.after(15000, begin_test_phase_transition)

def begin_practice():
    global current_phase, current_test
    current_phase = "practice"
    current_test = -1
    start_experiment()

def begin_test_phase_transition():
    global current_phase, current_test
    current_phase = "test"
    current_test = 0
    start_experiment()

def start_encoding_phase(image_path, shown_count, test_number_for_encoding):
    global shown_associated, shown_unassociated, shown_words
    associated_pool, unassociated_pool, _, _ = get_test_data(current_phase, test_number_for_encoding)

    num_associated_to_show = min(shown_count // 2, len(associated_pool))
    num_unassociated_to_show = min(shown_count // 2, len(unassociated_pool))

    shown_associated = random.sample(associated_pool, num_associated_to_show)
    shown_unassociated = random.sample(unassociated_pool, num_unassociated_to_show)
    shown_words = shown_associated + shown_unassociated
    random.shuffle(shown_words)

    reset_ui()
    photo = show_image(image_path)
    img_label = tk.Label(root, image=photo)
    img_label.image = photo
    img_label.pack(pady=10)

    frame = tk.Frame(root)
    frame.pack(pady=10)

    words_per_column = 5
    total_columns = (len(shown_words) + words_per_column - 1) // words_per_column
    for col in range(total_columns):
        for row in range(words_per_column):
            index = col * words_per_column + row
            if index < len(shown_words):
                word_label = tk.Label(frame, text=shown_words[index], font=("Times New Roman", 18))
                word_label.grid(row=row, column=col, padx=10, pady=10)

    root.after(30000, start_test_phase)

def reset_ui(include_test_labels=False):
    global prompt_label, instruction_label, instruction_label_widget
    for widget in root.winfo_children():
        widget.destroy()
    prompt_label = None
    instruction_label = None
    instruction_label_widget = None

    if include_test_labels:
        frame = tk.Frame(root)
        frame.pack(expand=True)

        prompt_label = tk.Label(frame, font=("Times New Roman", 26))
        prompt_label.pack(pady=(0, 20))

        instruction_label = tk.Label(frame, text="Left: NO   |   Right: YES\n\n Press escape to exit early.", font=("Times New Roman", 20))
        instruction_label.pack()

def start_test_phase():
    global test_words, additional_words_for_test, in_test_phase, current_phase
    in_test_phase = True
    reset_ui(include_test_labels=True)
    total_associated_test = 10
    min_shown_associated = 5

    associated_pool, _, _, _ = get_test_data(current_phase, current_test if current_test >= 0 else 0)
    max_shown_associated = min(len(shown_associated), total_associated_test)
    num_from_shown = random.randint(min_shown_associated, max_shown_associated)
    num_from_lure = total_associated_test - num_from_shown
    unseen_associated = list(set(associated_pool) - set(shown_associated))
    num_from_lure = min(num_from_lure, len(unseen_associated))
    num_from_shown = total_associated_test - num_from_lure
    associated_for_test = random.sample(shown_associated, num_from_shown)
    lure_associated = random.sample(unseen_associated, num_from_lure)
    non_associated_for_test = random.sample(shown_unassociated, 10)
    test_words = associated_for_test + lure_associated + non_associated_for_test
    random.shuffle(test_words)
    show_next_word()

def show_next_word():
    global current_index, prompt_label
    if current_index < len(test_words):
        if prompt_label:
            prompt_label.config(text=f"Did you see the word: {test_words[current_index]}?")
    else:
        save_results()
        global current_test, in_test_phase, current_phase
        in_test_phase = False
        if current_phase == "practice":
            current_phase = "intermission"
            start_experiment()
        else:
            current_test += 1
            if current_test < 3:
                start_experiment()
            else:
                reset_ui()
                label = tk.Label(root, text="Thank you! You have completed all tests.", font=("Times New Roman", 28))
                label.pack(expand=True)

def handle_response(event):
    global current_index, in_test_phase, prompt_label, response_data
    if not in_test_phase or current_index >= len(test_words):
        return
    if prompt_label is None:
        return
    word = test_words[current_index]
    was_shown = word in shown_words
    is_associated = word in shown_associated and word not in additional_words_for_test
    user_answer = event.keysym == 'Right'
    correct = user_answer == was_shown
    response_data.append({
        "Word": word,
        "WasShown": was_shown,
        "Associated": is_associated,
        "UserAnsweredYes": user_answer,
        "Correct": correct
    })
    current_index += 1
    show_next_word()

def save_results():
    global current_phase, current_test, participant_id, num_associated_shown_current_test, response_data
    if current_phase == "practice":
        response_data = []
        return

    if response_data:
        wb = Workbook()
        ws = wb.active
        ws.append(["Word", "WasShown", "Associated", "UserAnsweredYes", "Correct", "Associated and Correct"])
        for entry in response_data:
            associated_and_correct = entry["Associated"] and entry["Correct"]
            ws.append([
                entry["Word"],
                entry["WasShown"],
                entry["Associated"],
                entry["UserAnsweredYes"],
                entry["Correct"],
                associated_and_correct
            ])

        file_name = f"{participant_id} Test {current_test + 1} Results ({num_associated_shown_current_test} Associated Words).xlsx"
        file_path = os.path.join("C:/Users/fongj/OneDrive/Documents/Python Scripts/Personal Projects/Psychology Project/", file_name)
        try:
            wb.save(file_path)
        except Exception as e:
            print(f"Error saving results: {e}")

    response_data = []

# --- NEW: Escape handler ---
def handle_escape(event):
    global current_phase, response_data
    if current_phase != "practice" and response_data:
        save_results()
    root.destroy()

# --- Bindings and Start ---
root.bind("<Left>", handle_response)
root.bind("<Right>", handle_response)
root.bind("<Escape>", handle_escape)

start_experiment()
root.mainloop()
