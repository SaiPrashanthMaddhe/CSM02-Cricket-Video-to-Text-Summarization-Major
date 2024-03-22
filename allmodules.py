import torch
import os
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os.path, sys
import re
import csv
import pandas as pd
from paddleocr import PaddleOCR, draw_ocr # main OCR dependencies
from matplotlib import pyplot as plt # plot images
import cv2
import numpy as np
import os
from pickle import load
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model
from matplotlib import pyplot as plt
#pip install transformers[sentencepiece]
from transformers import BartTokenizer, BartForConditionalGeneration
import nltk
import random


from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline

import csv
import transformers
from transformers import pipeline

vgg_model = VGG16()
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

# extract features from each photo in the directory
def extract_features(filename, model):
    image = load_img(filename, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    return feature

# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text


def preprocess_images(filenames):
    images = [load_img(filename, target_size=(224, 224)) for filename in filenames]
    images = [img_to_array(image) for image in images]
    images = np.array(images)
    return preprocess_input(images)
def batch_extract_features(images, model):
    features = model.predict(images, verbose=0)
    return features

def test(directory):
    with open('tokenizer1.pkl', 'rb') as file:
        tokenizer = load(file)

    max_length = 25
    model = load_model('./Final_model.h5')
    lis = []
    with open('transcript.txt', 'w') as fobj:
        list1 = os.listdir(directory)
        batch_size = 8
        
        for i in range(0, len(list1), batch_size):
            batch_filenames = [os.path.join(directory, name) for name in list1[i:i+batch_size]]
            batch_images = preprocess_images(batch_filenames)
            batch_features = batch_extract_features(batch_images, vgg_model)

            for j, name in enumerate(list1[i:i+batch_size]):
                img = plt.imread(os.path.join(directory, name))
                plt.imshow(img)

                photo = batch_features[j:j+1]
                description = generate_desc(model, tokenizer, photo, max_length)
                description = ' '.join(description.split()[1:-1])
                fobj.write(f"Transcription: {description}\n")
                #plt.imshow(img)
                print(description)
                lis.append([i, description])
                

    print("Transcriptions Generated to transcript.txt!!")


    existing_data = pd.read_csv('cricket_data.csv')
    # Read the transcript data
    with open('transcript.txt', 'r') as file:
        transcription_lines = file.readlines()
    # Extract the transcription data
    transcription_data = pd.DataFrame(transcription_lines, columns=['transcription'])
    transcription_data['transcription'] = transcription_data['transcription'].str.extract(r"Transcription: (.*)")
    # Update the existing 'transcription' column with new data
    existing_data['transcription'] = transcription_data['transcription']
    # Save the updated data back to a CSV fileexisting_data.to_csv('cricket_data_with_updated_transcription.csv', index=False)
    existing_data.to_csv('cricket_datanew.csv', index=False)
    print("Transcriptions appended to cricket_datanew.csv!!")


def generate_gameplay_summary(data):
    import pandas as pd
    import random

    # Read the CSV file
    data = pd.read_csv("cricket_datanew.csv")

    # Open the file for writing
    with open("gameplay_sentences.txt", "w") as file:

        # Define positive, neutral, and negative templates (assuming they are defined in the code)
        negative_templates = {
        "l1_templates": (
        "{team_name} is in a dire situation at {team_score} for {wickets} in {overs} overs.",
        "{team_name} is struggling at {team_score} runs with {wickets} wickets down after {overs} overs.",
        "Facing the fierce pace, {team_name} is barely holding on at {team_score} runs with {wickets} wickets down in {overs} overs.",
        "{team_name} is facing a tough challenge, standing at {team_score} runs for the loss of {wickets} wickets after {overs} overs.",
        "{team_name} is in trouble, managing only {team_score} runs while losing {wickets} wickets in {overs} overs.",
        "{team_name} is grappling with a daunting situation at {team_score} runs with {wickets} wickets down in {overs} overs.",
        "{team_name} is under immense pressure at {team_score} runs with {wickets} wickets down after {overs} overs.",
        "Amidst the onslaught of the opposition, {team_name} is struggling at {team_score} runs with {wickets} wickets down in {overs} overs.",
        "With the match slipping away, {team_name} finds itself at {team_score} runs with {wickets} wickets down after {overs} overs.",
        "{team_name} is facing an uphill battle at {team_score} runs with {wickets} wickets down after {overs} overs."
        ),
        "l2_templates": (
        "{striker_name} is struggling with {striker_runs} runs off {striker_balls} balls and is confronting {bowler_name}, who has taken {bowler_wickets} wickets conceding {bowler_runs} runs.",
        "{striker_name} is facing the fierce pace of {bowler_name}, having scored {striker_runs} runs off {striker_balls} balls, while {bowler_name} has {bowler_wickets} wickets for {bowler_runs} runs.",
        "Under the pressure of fiery spells, {striker_name} has managed only {striker_runs} runs off {striker_balls} balls, facing {bowler_name}, who has {bowler_wickets} wickets for {bowler_runs} runs.",
        "{striker_name} is barely holding on with {striker_runs} runs off {striker_balls} balls, trying to tackle {bowler_name}, who has {bowler_wickets} wickets for {bowler_runs} runs.",
        "Struggling against {bowler_name}'s pace, {striker_name} has scored {striker_runs} runs off {striker_balls} balls, whereas {bowler_name} has {bowler_wickets} wickets for {bowler_runs} runs.",
        "{striker_name} is finding it hard to settle against {bowler_name}, managing only {striker_runs} runs off {striker_balls} balls, while the bowler has {bowler_wickets} wickets for {bowler_runs} runs.",
        "{striker_name} is under immense pressure, scoring {striker_runs} runs off {striker_balls} balls against {bowler_name}, who has {bowler_wickets} wickets for {bowler_runs} runs.",
        "Struggling to find rhythm, {striker_name} is battling {bowler_name}'s aggression, with {striker_runs} runs off {striker_balls} balls, while the bowler has {bowler_wickets} wickets for {bowler_runs} runs.",
        "{striker_name} is facing a formidable challenge against {bowler_name}, managing only {striker_runs} runs off {striker_balls} balls, with the bowler having {bowler_wickets} wickets for {bowler_runs} runs.",
        "With {striker_runs} runs off {striker_balls} balls, {striker_name} is struggling to handle {bowler_name}, who has {bowler_wickets} wickets for {bowler_runs} runs."
        ),
        "l3_templates": (
        "{bowler_name} is delivering the ball at a menacing speed of {speed}.",
        "The speed gun registers {speed} for {bowler_name}'s deliveries, adding to the pressure.",
        "Bowling with fiery pace, {bowler_name} is consistently clocking {speed}.",
        "{bowler_name} is putting the batsmen under immense pressure with deliveries reaching {speed}.",
        "The batsmen are struggling to cope with {bowler_name}'s speed, which is recorded at {speed}.",
        "{bowler_name} is unleashing a fierce attack, with deliveries clocking {speed}.",
        "With {speed} kmph deliveries, {bowler_name} is making life difficult for the batsmen.",
        "The opposition bowler, {bowler_name}, is maintaining an intimidating pace of {speed}.",
        "Delivering with sheer pace, {bowler_name} is proving to be a handful for the batsmen, with speeds reaching {speed}.",
        "{bowler_name} is bowling with savage intensity, making it hard for the batsmen to settle, with speeds touching {speed}."
        ),

        "l4_templates": (
        "With a dismal run rate of {runrate}, {team_name} urgently needs to achieve a run rate of {reqrun} to have any hope of winning.",
        "{team_name} is in desperate need of a run rate of {reqrun} to turn the tide, given their current run rate of {runrate}.",
        "To salvage the match, {team_name} requires a run rate of {reqrun} more, with the current run rate languishing at {runrate}.",
        "Facing a daunting task, {team_name} needs to achieve a run rate of {reqrun} for victory, struggling with a run rate of {runrate}.",
        "The chances look bleak for {team_name} with a run rate of {runrate}, needing to achieve a run rate of {reqrun} to even compete.",
        "{team_name} is desperately seeking a run rate of {reqrun} to counter the relentless attack of the opposition, with a run rate of {runrate}.",
        "With a run rate of {runrate}, {team_name} needs to achieve a run rate of {reqrun} to mount a comeback against the fierce opposition.",
        "The match hangs in the balance for {team_name}, as they require a run rate of {reqrun} to stay alive in the face of the opposition's onslaught, with a run rate of {runrate}.",
        "{team_name} is struggling to keep up with the required run rate, needing to achieve a run rate of {reqrun} to challenge the opposition's dominance.",
        "With a {reqrun} run rate needed for victory, {team_name} faces an uphill battle against the opposition's relentless attack, with a run rate of {runrate}."
        ),

        "l5_templates": (
        "The situation is grim, with {action}.",
        "{action} describes the dire state of affairs in the match.",
        "The ongoing struggle can be summed up as {action}.",
        "{action} encapsulates the immense challenge faced by {team_name}.",
        "Amidst the fiery spells, {action} prevails.",
        "{action} characterizes the intense battle unfolding in the match.",
        "The match hangs in the balance, with {action} adding to the tension.",
        "{action} reflects the uphill battle faced by {team_name} against the relentless opposition.",
        "The atmosphere is tense, with {action} defining the course of the match.",
        "The match is on a knife-edge, with {action} dictating the outcome."
        ),

        "l6_templates": (
        "And {transcription}, highlighting the difficult situation.",
        "Additionally, {transcription}, emphasizing the tough challenge ahead.",
        "Furthermore, {transcription}, underlining the struggle faced by {team_name}.",
        "In addition, {transcription}, reflecting the dire circumstances.",
        "Moreover, {transcription}, signifying the uphill battle for {team_name}.",
        "The commentary echoes the challenges faced by {team_name} in the ongoing battle.",
        "In the commentary, {transcription} emphasizes the intensity of the match.",
        "{transcription} captures the mood of desperation among {team_name} supporters.",
        "{transcription} underscores the need for a dramatic turnaround for {team_name} to emerge victorious.",
        "As per the commentary, {transcription} symbolizes the determination of {team_name} in the face of adversity."
        )
        }
        positive_templates = {
        "l1_templates": (
        "{team_name} is anchoring the innings at {team_score} for {wickets} in {overs} overs.",
        "{team_name} is leading the charge at {team_score} runs with {wickets} wickets down after {overs} overs.",
        "With {team_name} holding the fort at {team_score} runs with {wickets} wickets down in {overs} overs, the batting display is formidable.",
        "{team_name} is showcasing resilience at {team_score} runs with {wickets} wickets down, anchoring the innings in {overs} overs.",
        "{team_name} is standing strong at {team_score} runs for the loss of {wickets} wickets after {overs} overs, the performance is commendable.",
        "With a valiant effort, {team_name} stands at {team_score} runs with {wickets} wickets down in {overs} overs.",
        "{team_name} is dominating the proceedings at {team_score} runs with {wickets} wickets down after {overs} overs.",
        "{team_name} is setting a strong foundation at {team_score} runs with {wickets} wickets down in {overs} overs, showcasing determination.",
        "{team_name} is putting up a stellar performance at {team_score} runs with {wickets} wickets down after {overs} overs.",
        "Amidst a valiant effort by {team_name}, they stand at {team_score} runs with {wickets} wickets down in {overs} overs."
        ),
        "l2_templates": (
        "{striker_name} has played a valiant innings, scoring {striker_runs} runs off just {striker_balls} balls, anchoring the batting display.",
        "{striker_name} has shown impressive determination, knocking {striker_runs} runs off just {striker_balls} balls, holding the fort for {team_name}.",
        "With an impressive effort, {striker_name} has scored {striker_runs} runs off just {striker_balls} balls, showcasing resilience for {team_name}.",
        "{striker_name} is anchoring the innings with a valiant display, scoring {striker_runs} runs off just {striker_balls} balls for {team_name}.",
        "{striker_name} is leading the charge for {team_name} with an impressive {striker_runs} runs off just {striker_balls} balls, showing immense determination.",
        "{striker_name} is spearheading the batting attack for {team_name}, scoring {striker_runs} runs off just {striker_balls} balls with immense determination.",
        "{striker_name} is showcasing resilience at the crease, with an impressive {striker_runs} runs off just {striker_balls} balls for {team_name}.",
        "{striker_name} is holding the fort for {team_name}, with a valiant {striker_runs} runs off just {striker_balls} balls.",
        "{striker_name} is putting up a stellar performance, scoring {striker_runs} runs off just {striker_balls} balls, leading the batting charge for {team_name}.",
        "{striker_name} is demonstrating exceptional skill, scoring {striker_runs} runs off just {striker_balls} balls, to anchor the innings for {team_name}."
        ),
        "l3_templates": (
        "{bowler_name} is struggling to contain {team_name}'s formidable batting display, with deliveries registering a pace of {speed} and having claimed {bowler_wickets} wickets.",
        "{bowler_name} is facing a tough challenge against {team_name}'s aggressive batting lineup, with {bowler_wickets} wickets to his name and delivering balls at {speed}.",
        "The opposition bowler, {bowler_name}, is finding it hard to stem the flow of runs against {team_name}, despite {bowler_wickets} wickets and a pace of {speed}.",
        "With {team_name} on the attack, {bowler_name} is under immense pressure to deliver for the opposition, despite having taken {bowler_wickets} wickets and bowling at {speed}.",
        "{bowler_name} is grappling with {team_name}'s aggressive approach, struggling to make an impact, despite {bowler_wickets} wickets and a pace of {speed}.",
        "{bowler_name} is finding it hard to make inroads against {team_name}'s resilient batting lineup, despite bowling at {speed} and having claimed {bowler_wickets} wickets.",
        "{bowler_name} is facing an uphill battle against {team_name}'s batting onslaught, with {bowler_wickets} wickets in hand and delivering balls at {speed}.",
        "The opposition bowler, {bowler_name}, is under immense pressure to break {team_name}'s strong partnership, despite having taken {bowler_wickets} wickets and bowling at {speed}.",
        "{bowler_name} is finding it challenging to bowl against {team_name}'s well-set batsmen, despite having claimed {bowler_wickets} wickets and delivering balls at {speed}",
        "{bowler_name} is struggling to find a breakthrough against {team_name}'s formidable batting lineup, despite bowling at {speed} kmph and having taken {bowler_wickets} wickets.",
        ),
        "l4_templates": (
        "{team_name} is maintaining an impressive run rate of {runrate}, setting a strong foundation for victory.",
        "With a commendable run rate of {runrate}, {team_name} is dominating the proceedings.",
        "{team_name} is setting the tempo of the match with an impressive run rate of {runrate}.",
        "The run rate for {team_name} is exceptional at {runrate}, showcasing their intent for victory.",
        "With a formidable run rate of {runrate}, {team_name} is in the driver's seat, dictating terms to the opposition.",
        "{team_name} is dictating the pace of the match with a dominant run rate of {runrate}, putting the opposition on the back foot.",
        "The run rate for {team_name} is impressive at {runrate}, reflecting their strong position in the match.",
        "{team_name} is maintaining a brisk run rate of {runrate}, keeping the scoreboard ticking in their favor.",
        "With an aggressive run rate of {runrate}, {team_name} is firmly in control of the match.",
        "{team_name} is putting up a commanding performance with a formidable run rate of {runrate}."
        ),
        "l5_templates": (
        "The atmosphere is electric, with {action} driving {team_name} towards victory.",
        "{action} signifies {team_name}'s dominance in the match.",
        "With {action}, {team_name} is firmly in control of the proceedings.",
        "The ongoing battle can be summarized as {action}, highlighting {team_name}'s stronghold.",
        "Amidst the cheers of the crowd, {action} boosts {team_name}'s confidence.",
        "{action} epitomizes {team_name}'s relentless pursuit of victory.",
        "The match is tilting in favor of {team_name} with {action} shaping the outcome.",
        "{action} reflects the resilience of {team_name} in the face of adversity.",
        "The momentum is firmly with {team_name}, with {action} driving their charge.",
        "{action} underscores {team_name}'s determination to emerge victorious."
        ),
        "l6_templates": (
        "Moreover, {transcription}, highlighting {team_name}'s dominant performance.",
        "In addition, {transcription}, emphasizing {team_name}'s control over the match.",
        "Furthermore, {transcription}, underlining {team_name}'s relentless pursuit of victory.",
        "Additionally, {transcription}, reflecting {team_name}'s strong position in the match.",
        "{transcription}, as per the commentary, signifies {team_name}'s authoritative display.",
        "{transcription} captures the essence of {team_name}'s performance in the ongoing battle.",
        "As per the commentary, {transcription} symbolizes {team_name}'s quest for victory.",
        "{transcription} reflects the jubilant mood among {team_name} supporters.",
        "Moreover, {transcription} highlights {team_name}'s unwavering determination.",
        "The commentary echoes {transcription}, emphasizing {team_name}'s dominance."
        )
        }
        neutral_templates = {
        "l1_templates": (
        "{team_name} is at {team_score} for {wickets} in {overs} overs.",
        "{team_name} has scored {team_score} runs with {wickets} wickets down after {overs} overs.",
        "At {team_score} runs for the loss of {wickets} wickets in {overs} overs, {team_name} is steady.",
        "{team_name} stands at {team_score} runs with {wickets} wickets down after {overs} overs.",
        "With {team_score} runs and {wickets} wickets down in {overs} overs, {team_name} is holding its ground.",
        "{team_name} is holding steady at {team_score} runs with {wickets} wickets down in {overs} overs.",
        "At {team_score} runs with {wickets} wickets down after {overs} overs, {team_name} is maintaining composure.",
        "{team_name} is at {team_score} runs with {wickets} wickets down after {overs} overs, showing resilience.",
        "With {team_score} runs and {wickets} wickets down in {overs} overs, {team_name} is showing determination.",
        "{team_name} has reached {team_score} runs with {wickets} wickets down in {overs} overs, consolidating its position."
        ),
        "l2_templates": (
        "{striker_name} has scored {striker_runs} runs off {striker_balls} balls, anchoring the innings for {team_name}.",
        "{striker_name} is contributing with {striker_runs} runs off {striker_balls} balls, supporting {team_name}'s batting display.",
        "With {striker_runs} runs off {striker_balls} balls, {striker_name} is holding the fort for {team_name}.",
        "{striker_name} is playing a steady innings, scoring {striker_runs} runs off {striker_balls} balls for {team_name}.",
        "{striker_name} is consolidating the innings with {striker_runs} runs off {striker_balls} balls for {team_name}.",
        "{striker_name} is building a partnership, scoring {striker_runs} runs off {striker_balls} balls for {team_name}.",
        "{striker_name} is showing resilience at the crease, with {striker_runs} runs off {striker_balls} balls for {team_name}.",
        "{striker_name} is steadying the ship for {team_name}, scoring {striker_runs} runs off {striker_balls} balls.",
        "{striker_name} is contributing valuable runs, scoring {striker_runs} runs off {striker_balls} balls for {team_name}.",
        "{striker_name} is holding firm, with {striker_runs} runs off {striker_balls} balls, providing stability for {team_name}."
        ),
        "l3_templates": (
        "{bowler_name} is maintaining a disciplined line and length against {team_name}'s batting lineup, consistently clocking {speed} kmph and having taken {bowler_wickets} wickets.",
        "{bowler_name} is bowling with accuracy, probing {team_name}'s batsmen for a breakthrough, with speeds touching {speed} kmph and {bowler_wickets} wickets in hand.",
        "{bowler_name} is keeping it tight against {team_name}'s batting lineup, conceding minimal runs while bowling at {speed} kmph and having claimed {bowler_wickets} wickets.",
        "{bowler_name} is applying pressure on {team_name}'s batsmen with consistent line and length, bowling at {speed} kmph and already having taken {bowler_wickets} wickets.",
        "{bowler_name} is keeping {team_name}'s batsmen on their toes with a variety of deliveries, maintaining {speed} kmph and having taken {bowler_wickets} wickets.",
        "{bowler_name} is showcasing skillful bowling against {team_name}'s batting lineup, mixing up pace and line well, with {speed} kmph deliveries and {bowler_wickets} wickets already.",
        "{bowler_name} is maintaining a good line and length, making it difficult for {team_name}'s batsmen to score freely, bowling consistently at {speed} kmph and having taken {bowler_wickets} wickets.",
        "{bowler_name} is bowling a tight spell against {team_name}, not giving away easy runs, with {speed} kmph deliveries and {bowler_wickets} wickets taken.",
        "{bowler_name} is keeping the runs in check against {team_name}'s batting lineup, bowling at {speed} kmph and already having taken {bowler_wickets} wickets.",
        "{bowler_name} is posing a challenge to {team_name}'s batsmen with disciplined bowling, maintaining {speed} kmph and having claimed {bowler_wickets} wickets.",
        ),
        "l4_templates": (
        "{team_name} is maintaining a steady run rate of {runrate} at this stage of the match.",
        "With a consistent run rate of {runrate}, {team_name} is building a solid foundation.",
        "{team_name} is pacing the innings well with a run rate of {runrate}, consolidating their position.",
        "The run rate for {team_name} is stable at {runrate}, indicating a controlled approach.",
        "{team_name} is playing sensibly, maintaining a run rate of {runrate} in this phase of the innings.",
        "With a steady run rate of {runrate}, {team_name} is keeping the scoreboard ticking.",
        "{team_name} is keeping it steady with a run rate of {runrate}, not taking unnecessary risks.",
        "The current run rate of {runrate} reflects {team_name}'s composed batting performance.",
        "{team_name} is playing cautiously, focusing on rotating the strike with a run rate of {runrate}.",
        "{team_name} is consolidating their position with a disciplined run rate of {runrate}."
        ),
        "l5_templates": (
        "The match is finely poised, with both teams engaging in a battle of nerves.",
        "The ongoing contest is a testament to the competitive spirit of the game.",
        "Both teams are evenly matched, setting the stage for an exciting finish.",
        "The atmosphere is tense, with the outcome of the match hanging in the balance.",
        "The match is delicately poised, with every run and wicket crucial in determining the result.",
        "The contest is heating up, with neither team willing to give an inch.",
        "It's a seesaw battle, with momentum swinging from one team to the other.",
        "The match is on a knife-edge, with the outcome uncertain till the very end.",
        "The tension is palpable, as both teams look to gain the upper hand.",
        "With the game finely balanced, every run and wicket is crucial in shaping the result."
        ),
        "l6_templates": (
        "Furthermore, {transcription}, highlighting the competitive nature of the match.",
        "In addition, {transcription}, emphasizing the importance of each moment in the game.",
        "Moreover, {transcription}, reflecting the intensity of the contest between the two teams.",
        "Additionally, {transcription}, underscoring the significance of every run and wicket.",
        "{transcription} captures the essence of the closely fought encounter between the two teams.",
        "{transcription} reflects the nail-biting finish expected in this thrilling match.",
        "As per the commentary, {transcription} symbolizes the unpredictability of the game.",
        "{transcription} highlights the drama unfolding in this fiercely contested match.",
        "Moreover, {transcription} signifies the passion and excitement synonymous with cricket.",
        "The commentary echoes {transcription}, showcasing the thrill of the sport."
        )
        }

        def generate_commentary(data):
            # Same code for generating commentary as before
            # Extract data from the input
            team_name = data['team_name']
            x = data['team_score']
            if pd.isna(x) or x == "":
                wickets = team_score = ""
            else:
                wickets, team_score = x.split('/') if '/' in x else ('', '')
            overs = data['overs']
            striker_name = data['striker_name']
            striker_runs = data['striker_runs']
            striker_balls = data['striker_balls']
            bowler_name = data['bowler_name']
            bowler_runs = data['bowler_runs']
            speed = data['speed']
            runrate = data['runrate']
            reqrun = data['reqrun']
            transcription = data['transcription']
            action = data['action']
            bowler_wickets= data['wickets']


            # Choose a random sentiment
            sentiment = random.choice(["positive", "neutral", "negative"])

            # Choose random templates based on sentiment
            if sentiment == "positive":
                l1_templates = positive_templates["l1_templates"]
                l2_templates = positive_templates["l2_templates"]
                l3_templates = positive_templates["l3_templates"]
                l4_templates = positive_templates["l4_templates"]
                l5_templates = positive_templates["l5_templates"]
                l6_templates = positive_templates["l6_templates"]
            elif sentiment == "neutral":
                l1_templates = neutral_templates["l1_templates"]
                l2_templates = neutral_templates["l2_templates"]
                l3_templates = neutral_templates["l3_templates"]
                l4_templates = neutral_templates["l4_templates"]
                l5_templates = neutral_templates["l5_templates"]
                l6_templates = neutral_templates["l6_templates"]
            else:
                l1_templates = negative_templates["l1_templates"]
                l2_templates = negative_templates["l2_templates"]
                l3_templates = negative_templates["l3_templates"]
                l4_templates = negative_templates["l4_templates"]
                l5_templates = negative_templates["l5_templates"]
                l6_templates = negative_templates["l6_templates"]

            # Choose random templates for each level
            l1_template = random.choice(l1_templates)
            l2_template = random.choice(l2_templates)
            l3_template = random.choice(l3_templates)
            l4_template = random.choice(l4_templates)
            l5_template = random.choice(l5_templates)
            l6_template = random.choice(l6_templates)

            # Fill templates with data
            l1_commentary = l1_template.format(
                team_name=team_name, team_score=team_score, wickets=wickets, overs=overs
            ) if not any(pd.isna([team_name, team_score, wickets, overs])) else ""

            l2_commentary = l2_template.format(
                striker_name=striker_name,
                striker_runs=striker_runs,
                striker_balls=striker_balls,
                bowler_name=bowler_name,
                wickets=wickets,
                bowler_runs=bowler_runs,
                team_name=team_name,
                bowler_wickets=bowler_wickets
            ) if not any(pd.isna([striker_name, striker_runs, striker_balls, bowler_name, wickets, bowler_runs, team_name,bowler_wickets])) else ""


            l3_commentary = l3_template.format(bowler_name=bowler_name, speed=speed, team_name=team_name, bowler_wickets= bowler_wickets) if not any(pd.isna([bowler_name, speed, team_name, bowler_wickets])) else ""


            l4_commentary = l4_template.format(team_name=team_name, runrate=runrate, reqrun=reqrun) if not any(pd.isna([team_name, runrate, reqrun])) else ""

            l5_commentary = l5_template.format(action=action, team_name=team_name) if not any(pd.isna([action, team_name])) else ""

            l6_commentary = l6_template.format(transcription=transcription, team_name=team_name) if not any(pd.isna([transcription, team_name])) else ""

            # Combine the commentaries
            commentary = f"{l1_commentary}{l2_commentary}{l3_commentary}{l4_commentary}{l5_commentary}{l6_commentary}"
            return commentary
                # Iterate through each row in the data and generate commentary
        for index, row in data.iterrows():
            commentary = generate_commentary(row)
            # Write commentary to the file
            file.write(commentary + "\n")  # Add a newline character after each commentary

    # File will automatically be closed when exiting the "with" block
        





def process_cricket_data():
    csv_file='cricket_datanew.csv'
    output_file='gameplay_sentences.txt'
    # Reading CSV data from a file using pd.read_csv
    cricket_data = pd.read_csv(csv_file)
    #fill_missing_values(cricket_data)
    generate_gameplay_summary(cricket_data)
    print(f"Gameplay sentences have been saved to {output_file}.")




def read_transcripts_from_file(file_path):
    with open(file_path, 'r') as file:
        transcripts = file.readlines()
    return transcripts

def remove_consecutive_duplicates(transcripts):
    filtered_transcripts = [transcripts[0]]  # Add the first transcript
    for i in range(1, len(transcripts)):
        if transcripts[i] != transcripts[i - 1]:  # Check if current transcript is different from the previous one
            filtered_transcripts.append(transcripts[i])
    return filtered_transcripts

def write_transcripts_to_file(transcripts, output_file_path):
    with open(output_file_path, 'w') as file:
        file.writelines(transcripts)

def summarygeneration():

        # Load tokenizer and model
    tokenizer = BartTokenizer.from_pretrained("./distilbart-cnn-12-6")
    model = BartForConditionalGeneration.from_pretrained("./distilbart-cnn-12-6")

    # Read input text file
    with open("gameplay_sentences.txt", "r") as file:
        file_content = file.read().strip()

    # Tokenize input text into sentences
    sentences = nltk.tokenize.sent_tokenize(file_content)

    # Maximum tokens in the longest sentence
    max_chunk_length = tokenizer.max_len_single_sentence - 2

    # Split sentences into chunks not exceeding max_chunk_length
    chunks = []
    chunk = ""
    length = 0
    for sentence in sentences:
        tokenized_sentence = tokenizer.tokenize(sentence)
        sentence_length = len(tokenized_sentence)

        if sentence_length > max_chunk_length:
            # Split long sentences into multiple chunks
            while sentence_length > 0:
                if sentence_length <= max_chunk_length:
                    chunks.append(' '.join(tokenized_sentence[:max_chunk_length]))
                    sentence_length = 0
                else:
                    chunks.append(' '.join(tokenized_sentence[:max_chunk_length]))
                    tokenized_sentence = tokenized_sentence[max_chunk_length:]
                    sentence_length = len(tokenized_sentence)
        else:
            combined_length = sentence_length + length
            if combined_length <= max_chunk_length:
                chunk += sentence + " "
                length = combined_length
            else:
                chunks.append(chunk.strip())
                chunk = sentence + " "
                length = sentence_length

    # Append remaining chunk
    if chunk.strip():
        chunks.append(chunk.strip())

    # Combine chunks into paragraphs
    paragraphs = []
    paragraph = ""
    for chunk in chunks:
        if len(paragraph.split()) < 100:  # Adjust the number of words per paragraph as needed
            paragraph += chunk + " "
        else:
            paragraphs.append(paragraph.strip())
            paragraph = chunk + " "
    if paragraph.strip():
        paragraphs.append(paragraph.strip())

    # Generate summaries for each paragraph
    with open("summary.txt", "w") as summary_file:
        for paragraph in paragraphs:
            inputs = tokenizer(paragraph, return_tensors="pt", max_length=1024, truncation=True)
            try:
                summary_ids = model.generate(**inputs)
                summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                summary_file.write(summary + "<br><br>")
            except IndexError:
                pass  # Skip the current input and continue with the next one
            

    




def video_to_frames(input):


    # Path to the input video file
    input_video_path = input

    # Directory to save the frames
    output_directory = "Frames/"

    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Open the video file
    video_capture = cv2.VideoCapture(input_video_path)

    # Check if the video is opened successfully
    if not video_capture.isOpened():
        print("Error opening video file")
        exit()

    # Initialize variables
    frame_count = 0

    try:
        # Loop through the video frames
        while True:
            # Read a frame from the video
            ret, frame = video_capture.read()
            name = './FRAMES/frame' + str(frame_count) + '.jpg'
            print('Creating...' + name)

            # Break the loop if no frame is retrieved
            if not ret:
                break
    # Save the frame
            frame_count += 1
            frame_filename = f'{output_directory}frame_{frame_count:04d}.jpg'  # Change format if needed (jpg, png, etc.)
            cv2.imwrite(frame_filename, frame)

            # Display the frame (optional)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred in video_to_frames: {e}")
    finally:
        # Release the video capture object and close any open windows
        video_capture.release()
        cv2.destroyAllWindows()
    return output_directory

def representative_frames(input):

    input_frames_dir = input
    # Input directory containing frames
    output_frames_dir = "R_frames1/"

    # Create output directory if it doesn't exist
    os.makedirs(output_frames_dir, exist_ok=True)

    try:
        # Initialize variables
        prev_frame = None
        threshold = 35  # Adjust this threshold as needed
        frame_count = 0

        # Loop through the input frames directory
        for filename in os.listdir(input_frames_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                # Read the frame
                frame_path = os.path.join(input_frames_dir, filename)
                frame = cv2.imread(frame_path)
                if frame is None:
                    continue

                # Convert frame to grayscale
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Calculate frame difference
                if prev_frame is not None:
                    frame_diff = cv2.absdiff(gray_frame, prev_frame)
                    difference = cv2.mean(frame_diff)[0]

                    # If difference is below threshold, skip frame
                    if difference < threshold:
                        continue
                # Write the frame to the output directory with renamed file name
                output_frame_path = os.path.join(output_frames_dir, f"frame_{frame_count:05d}.jpg")
                cv2.imwrite(output_frame_path, frame)

                # Increment frame count
                frame_count += 1

                # Store current frame as previous frame for next iteration
                prev_frame = gray_frame.copy()

    except Exception as e:
        print(f"An error occurred in representative_frames: {e}")

    finally:
        print("Execution completed.")
    return output_frames_dir

def crop_framrs(input):

    IMAGES_DIR = input
    model_path = "best.pt"
    model = YOLO(model_path)
    threshold = 0.7

    OUTPUT_DIR = "cropped_frames/"

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    def crop_and_save_image(input_image_path, output_image_path, x1, y1, x2, y2):
        try:
            # Read the input image
            frame = cv2.imread(input_image_path)

            # Crop the image within the specified region
            roi = frame[int(y1):int(y2), int(x1):int(x2)]

            # Save the cropped image to the specified path
            cv2.imwrite(output_image_path, roi)

            print("Cropped image saved successfully at:", output_image_path)

        except Exception as e:
            print(f"Error occurred while processing {input_image_path}: {e}")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        # Iterate through the images in the input directory
        for image_file in os.listdir(IMAGES_DIR):
            if image_file.endswith('.jpg'):
                image_path = os.path.join(IMAGES_DIR, image_file)

                # Read the image
                frame = cv2.imread(image_path)

                # Perform object detection
                results = model(frame)[0]

                # Iterate through the detected objects
                for result in results.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = result

                    if score > threshold:
                        # Define output image path
                        output_image_path = os.path.join(OUTPUT_DIR, "cropped_" + image_file)

                        # Crop and save the image within the specified ROI
                        crop_and_save_image(image_path, output_image_path, x1, y1, x2, y2)

    except Exception as e:
        print(f"An error occurred: {e}")

    return OUTPUT_DIR


def text_extraxt_ocr(input):

    # Setup model
    ocr_model = PaddleOCR(lang='en', use_gpu=False) # need to run only once to download and load model into memory

    def ocr_on_folder(folder_path, output_file):
        try:
            filenames = sorted([filename for filename in os.listdir(folder_path) if filename.endswith(('.jpg', '.png', '.jpeg'))])
            with open(output_file, 'w', encoding='utf-8') as f:
                for filename in filenames:
                    try:
                        img_path = os.path.join(folder_path, filename)
                        frame_number = os.path.splitext(filename)[0].split('_')[-1]  # Extract frame number from filename
                        result = ocr_model.ocr(img_path)
                        f.write(f"Frame Number: {frame_number}\n")
                        write_strings(result, f)
                        f.write("\n\n")
                    except Exception as e:
                        print(f"Error occurred while processing {filename}: {e}")
        except Exception as e:
            print(f"Error occurred while opening or writing to the output file: {e}")

    def write_strings(result, file):
        for item in result:
            if isinstance(item, str):
                file.write(item + "\n")
            elif isinstance(item, list) or isinstance(item, tuple):
                write_strings(item, file)


    output="output_ocr.txt"

    # Call the function to perform OCR on all images in the "frames" folder
    ocr_on_folder(input, 'output_ocr.txt')

    return output


def text_formatting(input):

    text_output = "text_format.txt"

    # Combined regex patterns
    patterns = {
        "frame_no": r'^Frame Number: (\d{1,})$',
        "team": r'\b(?!RUN|REQ|OCR|SRI|REO)([A-Z]{2,3})([O0oa@e]?)?(\d+/\d+)?\b',
        "team_score": r'^\d+/\d+$',
        "striker": r'^(?=.*\*)(?!.*(?:NEED|TRAIL|TARGET|WIN))([A-Z]+(?:\s*[A-Z]*)*)\s*\*\s*(\d+)\s*(?:\((\d+)\))?',
        "non_striker": r'^(?!.*(?:NEED|TRAIL|TARGET|WIN))([A-Z]+(?:\s*[A-Z]*)*)\s*(\d+)\s*(?:\((\d+)\))?(?=\s|$)',
        "bowler": r'([A-Z]{4,})\s*(\d+)[/](\d+)',
        "overs": r'(?:OVERS\s*)?(\d{1,2}\.\d)(?!\s*KM/H)(?=\s|$)',
        "runrate": r'(RUN\s*RATE)\s*(\d+(\.\d+)?)',
        "reqrun": r'(REQ\.?\s*RATE)\s*(\d+(\.\d+)?)',
        "speed": r'(?:SPEED\s*)?(\d+(\.\d+)?\s*KM/H)',
        "action": r'(?:NEED|TRAIL|TARGET|WIN|BALLS)'
    }

    # Pre-compile regex patterns
    compiled_patterns = {key: re.compile(pattern) for key, pattern in patterns.items()}


# Function to extract information from a frame and write to file
    def extract_frame_info_and_write(frame_text, output_file, frame_number):
        values = {
            "frame_no": "N/A",
            "team_name": "N/A",
            "team_score": "N/A",
            "striker": {"striker_name": "N/A", "striker_runs": "N/A", "striker_balls": "N/A"},
            "non_striker": {"non_striker_name": "N/A", "non_striker_runs": "N/A", "non_striker_balls": "N/A"},
            "bowler": {"bowler_name": "N/A", "bowler_runs": "N/A", "wickets": "N/A"},
            "overs": "N/A",
            "runrate": "N/A",
            "reqrun": "N/A",
            "speed": "N/A",
            "action": "N/A"  # Initialize unmatched lines string
        }

        lines = frame_text.split('\n')

        for line in lines:
            for key, pattern in compiled_patterns.items():
                match = pattern.match(line)
                if match:
                    if key == "frame_no":
                        values["frame_no"] = match.group(1)  # Store frame number
                    elif key == "team":
                        values["team_name"] = match.group(1)
                        values["team_score"] = match.group(3) or "N/A"
                    elif key == "team_score":
                        values["team_score"] = match.group()
                    elif key == "striker":
                        values["striker"]["striker_name"] = match.group(1)
                        values["striker"]["striker_runs"] = match.group(2)
                        values["striker"]["striker_balls"] = match.group(3) or "N/A"
                    elif key == "non_striker":
                        values["non_striker"]["non_striker_name"] = match.group(1).strip()
                        values["non_striker"]["non_striker_runs"] = match.group(2)
                        values["non_striker"]["non_striker_balls"] = match.group(3) if match.group(3) else '0'
                    elif key == "bowler":
                        values["bowler"]["bowler_name"] = match.group(1)
                        values["bowler"]["bowler_runs"] = match.group(3)
                        values["bowler"]["wickets"] = match.group(2)
                    elif key == "overs":
                        values["overs"] = match.group(1)
                    elif key == "runrate":
                        values["runrate"] = match.group(2)
                    elif key == "reqrun":
                        values["reqrun"] = match.group(2)
                    elif key == "speed":
                        values["speed"] = match.group(1)
                elif re.search(patterns["action"],line):
                  values["action"]=line

        # Write values to file
        for key, value in values.items():
            output_file.write(f"{key}: {value}\n")
        output_file.write("\n")
        return values
    
    try:
        # Read input from file
        with open(input, 'r') as input_file, open(text_output, 'w') as output_file:
            input_text = input_file.read()

            # Split text into frames using empty lines as separators
            frames = input_text.split('\n\n')

            completed_frames = 0
            # Process each frame separately
            for i, frame in enumerate(frames, 1):
                try:
                    frame_info = extract_frame_info_and_write(frame, output_file, i)
                    completed_frames += 1
                    print(f"Frame {i} completed. Total completed frames: {completed_frames}")
                except Exception as e:
                    print(f"An error occurred while processing frame {i}: {e}")

        print("Frame processing completed.")

    except Exception as e:
        print(f"An error occurred: {e}")

    return text_output

def to_csv(input):

    # Define the input file name
    input_file = input

    # Define the output CSV file name
    output_file = "cricket_data.csv"

    # Initialize a list to store the cricket data
    cricket_data = []

    # Function to parse the input file and extract cricket data
    def parse_input_file(input_file):
        try:
            with open(input_file, "r") as file:
                current_frame = {}
                for line in file:
                    line = line.strip()
                    if line:
                        key, value = line.split(": ", 1)
                        if key.startswith("striker") or key.startswith("non_striker") or key.startswith("bowler"):
                            # Extract nested data from string and convert to dictionary
                            nested_data = eval(value)
                            # Update current frame with nested data
                            current_frame.update(nested_data)
                        else:
                            current_frame[key] = value
                    else:
                        cricket_data.append(current_frame)
                        current_frame = {}
        except Exception as e:
            print(f"Error occurred while parsing the input file: {e}")

    # Function to write cricket data to CSV file
    def write_to_csv(output_file):
        try:
            with open(output_file, "w", newline="") as csvfile:
                fieldnames = ["frame_no", "team_name", "team_score", "striker_name", "striker_runs", "striker_balls",
                              "non_striker_name", "non_striker_runs","non_striker_balls", "bowler_name", "bowler_runs", "wickets",
                              "overs", "runrate", "reqrun", "speed", "action"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                # Write header
                writer.writeheader()

                # Write cricket data
                for data in cricket_data:
                    writer.writerow(data)
        except Exception as e:
            print(f"Error occurred while writing to the CSV file: {e}")

    # Parse the input file
    parse_input_file(input_file)

    # Write cricket data to CSV file
    write_to_csv(output_file)
    print("Cricket data has been successfully stored in", output_file)
    return output_file


def perfect_csv(input):


    def remove_duplicate_rows(csv_file, ignore_column):
        try:
            df = pd.read_csv(csv_file)
            df_no_duplicates = df.drop_duplicates(subset=[col for col in df.columns if col != ignore_column])
            df_no_duplicates.to_csv(csv_file, index=False)
        except Exception as e:
            print(f"Error occurred while removing duplicate rows: {e}")

    def remove_consecutive_duplicates(csv_file, ignore_column):
        try:
            df = pd.read_csv(csv_file)
            df_no_consecutive_duplicates = df.drop_duplicates(subset=[col for col in df.columns if col != ignore_column])
            df_no_consecutive_duplicates.to_csv(csv_file, index=False)
        except Exception as e:
            print(f"Error occurred while removing consecutive duplicates: {e}")
"""
def crop(path1):
    path2 = "cropped_images/"
    os.makedirs(path2, exist_ok=True)
    listing = os.listdir(path1) 
    for item in listing:
        fullpath = os.path.join(path1,item) 
        newpath = os.path.join(path2, item)
        if os.path.isfile(fullpath):
            im = Image.open(fullpath)
            imCrop = im.crop((100, 625, 1200, 720))
            imCrop.save(newpath  , "BMP", quality=100)
    print("CROP NON YOLO COmpleted")
    return path2
"""
def crop(path1):
    def crop_and_save_bottom_half(image_path, output_path):
        # Read the image
        image = cv2.imread(image_path)
        
        # Get image dimensions
        height, width, _ = image.shape
        
        # Calculate midpoint for horizontal division
        midpoint = height // 2
        
        # Crop the bottom half of the image
        bottom_half = image[midpoint:, :]
        
        # Save the bottom half
        cv2.imwrite(output_path, bottom_half)

    # Input and output directories
    input_dir = path1
    output_dir = "output_images/"

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over each image file in the input directory
    for filename in os.listdir(input_dir):
        # Check if the file is an image
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            # Construct input and output paths
            input_image_path = os.path.join(input_dir, filename)
            output_image_path = os.path.join(output_dir, filename)
            
            # Perform cropping and saving the bottom half three times
            for i in range(3):
                if i == 0:
                    crop_and_save_bottom_half(input_image_path, output_image_path)
                else:
                    crop_and_save_bottom_half(output_image_path, output_image_path)
            

    print("Image processing completed!")
    return output_dir



def main():
    # Path to the input video file
    input = 'static/videos/cricket_video.mp4'
    # Directory to save the frames
    input2 = video_to_frames(input)
    print("Frames created")

    input3 = representative_frames(input2)
    print("representative frames created")

    """
    input4 = crop_framrs(input3)
    print("Frames are cropped")
    """

    inputx=crop(input3)


    input5 = text_extraxt_ocr(inputx)
    print("Text has been Extracted")

    print("started Text formatting!!")
    input6 = text_formatting(input5)
    print("Completed text formatting!!")

    input7 = to_csv(input6)

    test(input3)
    print("Transcripts appended to CSV Completed")

    input_csv='cricket_datanew.csv'
    input8 = perfect_csv(input7)
    print("perfect csv has been created")



    process_cricket_data()
    print("Processed CSV to Text file")

    

    output_file_path='summary.txt'
    transcripts = read_transcripts_from_file('gameplay_sentences.txt')
    #filtered_transcripts = remove_consecutive_duplicates(transcripts)

    #write_transcripts_to_file(filtered_transcripts, output_file_path)

    summarygeneration()
    print()
    print("SUMMARY COMPLETED!!!!!")
    
    

if __name__ == "__main__":
    main()

