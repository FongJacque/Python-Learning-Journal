import random
from PIL import Image, ImageDraw, ImageFont

def generate_bingo_card(pool):
    if len(pool) < 25:
        raise ValueError("Pool must contain at least 25 elements")
    card = [['' for _ in range(5)] for _ in range(5)]
    available_positions = [(row, col) for row in range(5) for col in range(5)]
    pool_copy = pool[:]
    for pos in available_positions:
        row, col = pos
        choice = random.choice(pool_copy)
        card[row][col] = choice
        pool_copy.remove(choice)
    return card

def create_bingo_image(card, filename="bingo_card.png"):
    width, height = 500, 550
    cell_size = width // 5
    background_color = (0, 0, 0)
    text_color = (255, 255, 255)
    border_color = (255, 255, 255)
    header_text = " B   I   N   G   O"
    img = Image.new('RGB', (width, height), background_color)
    draw = ImageDraw.Draw(img)
    # Optionally, load a font (default font if none is available)
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except IOError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), header_text, font=font)
    header_width = bbox[2] - bbox[0]
    header_x = (width - header_width) // 2
    header_y = 10
    draw.text((header_x, header_y), header_text, font=font, fill=text_color)
    for row in range(5):
        for col in range(5):
            x1, y1 = col * cell_size, row * cell_size + 50
            x2, y2 = (col + 1) * cell_size, (row + 1) * cell_size + 50
            draw.rectangle([x1, y1, x2, y2], outline=border_color, width=2)
            text = str(card[row][col])
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            text_x = x1 + (cell_size - text_width) / 2
            text_y = y1 + (cell_size - text_height) / 2
            draw.text((text_x, text_y), text, font=font, fill=text_color)
    img.save(filename)
    img.show()

def print_bingo_card(card):
    print(" B   I   N   G   O")
    for i in range(5):
        for j in range(5):
            print(f"{card[i][j]:2}", end="  ")
        print()

def bingo_game():
    pool_type = "numbers"  # Change this to "strings" to use a list of strings
    # If you want to use numbers, this remains as numbers from 1 to 75
    if pool_type == "numbers":
        pool = list(range(1, 76))
    # If you want to use strings, fill in the pool with your list of strings
    elif pool_type == "strings":
        pool = [
            'Apple', 'Banana', 'Cherry', 'Date', 'Elderberry',
            'Fig', 'Grapes', 'Honeydew', 'Kiwi', 'Lemon',
            'Mango', 'Nectarine', 'Orange', 'Papaya', 'Quince',
            'Raspberry', 'Strawberry', 'Tangerine', 'Uva', 'Watermelon',
            'Pineapple', 'Peach', 'Plum', 'Apricot', 'Blackberry'
        ]  # Add more strings as needed
    print("Welcome to Bingo!")
    bingo_card = generate_bingo_card(pool)
    print_bingo_card(bingo_card)
    create_bingo_image(bingo_card)
    print("\nGame Over!")

bingo_game()