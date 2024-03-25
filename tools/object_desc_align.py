def generate_prompt_from_file_name(file_name):
    '''
    Taking files name as input, generate object description in 
    "Complex scene with concealed *object name* in it" pattern.

    Since CAMO dataset do not have object name in file name, we use "Complex scene with camouflaged object in it." as prompt.
    '''
    if file_name.startswith("camourflage"):
        prompt = "Complex scene with camouflaged object in it."
    elif file_name.startswith("COD10K"):
        object_name = file_name.split("-")[5]
        prompt = f"Complex scene with concealed {object_name} in it."
    
    return prompt


# test

def main():
    file_name = "camourflage_00265.jpg"
    prompt = generate_prompt_from_file_name(file_name)
    print(prompt)

if __name__ == "__main__":
    main()