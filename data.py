import torch
from tqdm import tqdm

questions = [
        "Explain what rhythmic stabilization is for lower back pain.",
        "How should rhythmic stabilization exercises be performed for patients with lower back pain?",
        "Why is rhythmic stabilization beneficial for treating lower back pain?",
        "What muscles are involved in rhythmic stabilization for lower back pain?",
        "How frequently should rhythmic stabilization exercises be performed?",
        "What are the clinical outcomes of rhythmic stabilization therapy?",
        "How does rhythmic stabilization compare to other stabilization techniques?",
        "What are contraindications for rhythmic stabilization exercises?",
        "Describe the progression of rhythmic stabilization exercises.",
        "How can rhythmic stabilization be modified for patients of different ages?"
    ]

answers = [
    """Rhythmic stabilization is a proprioceptive neuromuscular facilitation (PNF) technique designed to improve trunk stability by inducing co‐contraction of the deep and superficial trunk muscles. In practice, it involves the therapist applying manual, multidirectional resistance while the patient performs isometric contractions of the trunk. This technique "trains" the neuromuscular system to maintain a stable spine during everyday movements, reducing excessive strain on the lower back.""",
    """The exercise is typically performed with the patient in a seated or supported position. The therapist initiates resistance against the patient's trunk flexion (or extension) with one hand while instructing the patient to hold a static contraction against this resistance. The resistance is then alternated – for example, shifting one hand from the anterior to the posterior aspect of the trunk – so that the patient continuously adapts to multidirectional forces. A common protocol involves:
        - Performing 15 repetitions per set,
        - Holding each contraction for about 8 seconds (with progression to longer holds over time),
        - Completing 3 sets per session,
        - With short rests between repetitions (around 30 seconds) and sets (about 1 minute),
        - And sessions lasting approximately 30 minutes, five times per week for several weeks.""",
    """Rhythmic stabilization is beneficial because it enhances neuromuscular control and improves the co-activation of antagonistic muscle groups. This leads to better spinal alignment, increased internal abdominal pressure, and improved load sharing between the trunk muscles. The result is a decrease in mechanical stress on the lumbar spine, reduced pain intensity, and improved functional performance. Studies have shown that patients engaging in RS experience significant reductions in pain and disability compared to conventional therapy alone.""", 
    """The muscles involved in RS include both the trunk flexors and extensors. Primarily, this means the deep stabilizers such as the transverse abdominis and multifidus, along with the rectus abdominis and oblique muscles, as well as the erector spinae group. By co-contracting these muscle groups, the exercise promotes a balanced stabilization of the lumbar spine.""",
    """In clinical studies, RS exercises are frequently performed five times per week over a four‐week period. The regimen is structured into sessions that last around 30 minutes. The frequency, however, can be adjusted based on patient tolerance, severity of symptoms, and overall fitness level; some patients might start with three sessions per week before progressing to a more intensive schedule.""", 
    """Clinical outcomes associated with rhythmic stabilization include:
        - Significant reduction in pain intensity (as measured by Visual Analogue Scale scores),
        - Decreased disability (using indices like the Oswestry Disability Index),
        - Improved trunk muscle endurance,
        - Enhanced spinal mobility and postural control. These improvements translate into better functional capacity in daily activities and a reduced risk of further injury.""",
    """When compared to other stabilization techniques, RS offers the advantage of simultaneous activation of both flexor and extensor muscles through alternating isometric contractions. This bilateral, multidirectional approach may lead to a more balanced neuromuscular response than isolated strengthening exercises. Some research also suggests that combining RS with isotonic exercises can further enhance outcomes, indicating that RS might provide superior improvements in core stability and pain reduction relative to conventional methods.""",
    """Contraindications for RS include conditions where additional spinal loading or resistance could be harmful. These may involve:
        - Acute nerve root pain or severe radiculopathy,
        - Structural spinal pathologies (e.g., spondylolisthesis, severe spondylosis, recent vertebral fractures),
        - Systemic conditions such as active infections (e.g., spinal tuberculosis) or severe osteoporosis,
        - Post-spinal surgery patients who have not yet regained stability. Patients should be carefully screened, and modifications or alternative approaches should be considered if any contraindicating conditions exist.""",
    """Progression in RS exercises is typically gradual. For example, a study protocol might begin with:
        - Weeks 1–2: 3 sets of 15 repetitions with an 8-second hold per contraction,
        - Week 3: Progressing to 4 sets with a 10-second hold,
        - Week 4: Advancing to 5 sets with a 12-second hold. This progression allows for increasing the duration and workload as the patient's strength and neuromuscular control improve. Resistance can also be adjusted by modifying the force applied by the therapist or by altering the patient's position to increase difficulty.""",
    """Modifications for different age groups or fitness levels are essential:
        - For older patients or those with lower baseline strength, the intensity (i.e., the level of resistance and duration of holds) should be reduced initially. They might start with fewer repetitions or shorter hold times, gradually increasing as their tolerance improves.
        - For younger or more physically fit individuals, progression can be more aggressive by increasing hold times, the number of sets, or by introducing more challenging variations (e.g., unstable surfaces or reduced support).
        - It is also important to ensure that all modifications maintain proper alignment and safety to avoid undue stress on the spine.""",
]

# Custom Dataset class
class PhysioDataset(torch.utils.data.Dataset):
    def __init__(self, questions, answers, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.inputs = []
        
        print("Tokenizing dataset...")
        for question, answer in tqdm(zip(questions, answers), total=len(questions)):
            # Format with Qwen2 chat template
            text = (
                f"<|im_start|>system\nYou are a knowledgeable physiotherapy assistant that provides accurate and helpful information about physiotherapy techniques and treatments.\n<|im_end|>\n"
                f"<|im_start|>user\n{question}\n<|im_end|>\n"
                f"<|im_start|>assistant\n{answer}\n<|im_end|>"
            )
            
            # Tokenize
            tokenized = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            
            # Store as PyTorch tensors
            self.inputs.append({
                "input_ids": tokenized["input_ids"][0],
                "attention_mask": tokenized["attention_mask"][0]
            })
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        item = self.inputs[idx]
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "labels": item["input_ids"].clone()  # For causal LM, labels are the same as inputs
        }
    
