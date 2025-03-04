import transformers

def generate_assessment(
   age, gender, occupation, activity_level, other_comments,
   pain_location, pain_intensity, pain_duration, pain_character,
   aggravating_factors, relieving_factors, previous_injuries,
   chronic_conditions, medications,
   functional_limitations, patient_goals, equipment_at_home):
  
   model_name = "Qwen/Qwen2-0.5B"
   model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
   tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
   pipe = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer)
   chat_history = [
   {"role": "system", "content": "You are an AI assistant specialized in physiotherapy assessments. Your task is to conduct an initial assessment for me using the provided information. Analyze the data and generate a concise summary of my condition, potential diagnoses, and recommendations for further evaluation or treatment."},
   {"role": "user"  , "content": f''' Use the following information to conduct your assessment:
   1. My Demographic Data:
      Age: {age}
      Gender: {gender}
      Occupation: {occupation}
      Activity Level: {activity_level}
   2. My Pain Information:
      Location: {pain_location}
      Intensity (0-10 scale): {pain_intensity}
      Duration: {pain_duration}
      Character (e.g., sharp, dull, aching): {pain_character}
      Aggravating Factors: {aggravating_factors}
      Relieving Factors: {relieving_factors}
      Other Comments:  {other_comments}
   3. My Medical History:
      Previous Injuries: {previous_injuries}
      Chronic Conditions: {chronic_conditions}
      Medications: {medications}
   4. Personalizations and Goals:
      My Functional Limitations: {functional_limitations}
      My Goals: {patient_goals}
      Equipment Owned: {equipment_at_home}
     
   Assessment Report:
  
   Based on the information above, generate an assessment report for the following:
   1. A summary of the conditions (2-3 sentences)
   2. Any red flags or concerns that require immediate medical attention (if applicable). If not applicable say no immediate red flags or concerns.
   3. A table of diagnoses and the probability (high, medium, low) of each diagnoses (list 2-3 possibilities).
   4. Recommendations for further evaluation that could be done at home and how to interpret the evaluation results (3-4 points).
   5. Suggest 4-5 personalized treatment plan based on the personalization and goals section, such as recommended exercises, rehab activities. Suggest the number of reps and weights if applicable. Rank activities by most recommended (high to low).
   Present your assessment in a clear, concise manner suitable for both me and a physiotherapist to review. Remember to maintain a professional tone and emphasize that this is an initial assessment based on provided information, not a definitive diagnosis.
  
   Generate the text in nicely formatted Markdown, using bold and tables where necessary to make the assessment more clear.'''}
   ]
   print(chat_history)
   outputs = pipe(chat_history, max_new_tokens=256)


   return outputs[0]["generated_text"][-1]['content']


if __name__ == "__main__":
   age = 30
   gender = "Male"
   occupation = "Software Engineer"
   activity_level = "moderate"
   other_comments = "I have been experiencing pain for the past 4 months"
   pain_location = "lower back"
   pain_intensity = 5
   pain_duration = "chronic"
   pain_character = "midly sharp"
   aggravating_factors = "sitting on sofa"
   relieving_factors = "changing sitting location"
   previous_injuries = "Fall from stairs"
   chronic_conditions = "None"
   medications = "Ibuprofen gel"
   functional_limitations = "Difficulty sitting on sofa for long hours"
   patient_goals = "To not feel pain while sitting"
   equipment_at_home = "Weights, yoga mat"

   print(generate_assessment(age, gender, occupation, activity_level, other_comments, pain_location, pain_intensity, pain_duration, pain_character, aggravating_factors, relieving_factors, previous_injuries, chronic_conditions, medications, functional_limitations, patient_goals, equipment_at_home))