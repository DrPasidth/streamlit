import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import math

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class CombinedHealthAnalyzer:
    def __init__(self):
        # Initialize MediaPipe models
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Complete palm health interpretations (expanded from previous version)
        self.palm_health_interpretations = {
            'heart_line': {
                'long': {
                    'en': "Strong emotional nature, passionate in relationships. Traditional palmistry suggests good cardiovascular health. Consider heart-healthy activities like cardiovascular exercise.",
                    'th': "มีอารมณ์แรงกล้า หลงใหลในความสัมพันธ์ ตามหลักการดูลายมือแบบดั้งเดิม แสดงถึงสุขภาพหัวใจและหลอดเลือดที่ดี ควรพิจารณากิจกรรมที่ดีต่อหัวใจ เช่น การออกกำลังกายแบบคาร์ดิโอ"
                },
                'medium': {
                    'en': "Balanced emotional expression, stable relationships. Traditional palmistry suggests moderate cardiovascular health. Regular relaxation techniques may support heart wellness.",
                    'th': "แสดงอารมณ์อย่างสมดุล ความสัมพันธ์มั่นคง ตามหลักการดูลายมือแบบดั้งเดิม แสดงถึงสุขภาพหัวใจและหลอดเลือดปานกลาง เทคนิคการผ่อนคลายเป็นประจำอาจช่วยสนับสนุนสุขภาพหัวใจ"
                },
                'short': {
                    'en': "Reserved emotional nature, selective in relationships. Traditional palmistry suggests need for heart health attention. Consider stress management practices for cardiovascular wellness.",
                    'th': "สงวนอารมณ์ เลือกสรรในความสัมพันธ์ ตามหลักการดูลายมือแบบดั้งเดิม แสดงถึงความต้องการดูแลสุขภาพหัวใจ ควรพิจารณาการฝึกจัดการความเครียดเพื่อสุขภาพหัวใจและหลอดเลือด"
                }
            },
            'head_line': {
                'long': {
                    'en': "Analytical mind, good problem-solving abilities. Traditional palmistry suggests excellent mental clarity and cognitive function. Consider brain-healthy activities like reading and puzzles.",
                    'th': "จิตใจวิเคราะห์ ความสามารถในการแก้ปัญหาดี ตามหลักการดูลายมือแบบดั้งเดิม แสดงถึงความชัดเจนทางจิตใจและการทำงานของสมองที่ยอดเยี่ยม ควรพิจารณากิจกรรมที่ดีต่อสมอง เช่น การอ่านและปริศนา"
                },
                'medium': {
                    'en': "Practical thinking, balanced decision making. Traditional palmistry suggests good mental health. Regular mental exercises may help maintain cognitive function.",
                    'th': "การคิดเชิงปฏิบัติ การตัดสินใจที่สมดุล ตามหลักการดูลายมือแบบดั้งเดิม แสดงถึงสุขภาพจิตที่ดี การออกกำลังกายทางจิตใจเป็นประจำอาจช่วยรักษาการทำงานของสมอง"
                },
                'short': {
                    'en': "Intuitive thinking, creative approach to problems. Traditional palmistry suggests need for mental wellness support. Consider meditation and mindfulness practices.",
                    'th': "การคิดเชิงสัญชาตญาณ แนวทางสร้างสรรค์ในการแก้ปัญหา ตามหลักการดูลายมือแบบดั้งเดิม แสดงถึงความต้องการสนับสนุนสุขภาพจิต ควรพิจารณาการทำสมาธิและการฝึกสติ"
                }
            },
            'life_line': {
                'long': {
                    'en': "Strong vitality, adventurous spirit. Traditional palmistry suggests excellent physical constitution and energy levels. Focus on maintaining balanced nutrition and regular exercise.",
                    'th': "พลังชีวิตแข็งแกร่ง จิตวิญญาณผจญภัย ตามหลักการดูลายมือแบบดั้งเดิม แสดงถึงร่างกายที่แข็งแรงและระดับพลังงานที่ยอดเยี่ยม ควรเน้นการรับประทานอาหารที่สมดุลและออกกำลังกายอย่างสม่ำเสมอ"
                },
                'medium': {
                    'en': "Good health awareness, balanced lifestyle. Traditional palmistry suggests moderate vitality. Consider regular physical activity and stress management techniques.",
                    'th': "ตระหนักถึงสุขภาพดี วิถีชีวิตสมดุล ตามหลักการดูลายมือแบบดั้งเดิม แสดงถึงพลังชีวิตปานกลาง ควรพิจารณาการออกกำลังกายอย่างสม่ำเสมอและเทคนิคการจัดการความเครียด"
                },
                'short': {
                    'en': "Careful nature, prefers stability and routine. Traditional palmistry suggests need for energy conservation. Consider gentle exercise like yoga or tai chi and adequate rest.",
                    'th': "ธรรมชาติระมัดระวัง ชอบความมั่นคงและกิจวัตร ตามหลักการดูลายมือแบบดั้งเดิม แสดงถึงความต้องการอนุรักษ์พลังงาน ควรพิจารณาการออกกำลังกายเบาๆ เช่น โยคะหรือไทชิ และการพักผ่อนที่เพียงพอ"
                }
            },
            'mercury_line': {
                'present': {
                    'en': "Traditional palmistry suggests good digestive health and metabolism. Consider maintaining a balanced diet rich in fiber and staying well-hydrated.",
                    'th': "ตามหลักการดูลายมือแบบดั้งเดิม แสดงถึงสุขภาพระบบย่อยอาหารและการเผาผลาญที่ดี ควรรักษาอาหารที่สมดุลและมีเส้นใยสูง รวมทั้งดื่มน้ำให้เพียงพอ"
                },
                'absent': {
                    'en': "Traditional palmistry suggests paying attention to digestive health. Consider regular meals, proper hydration, and foods that support digestive wellness.",
                    'th': "ตามหลักการดูลายมือแบบดั้งเดิม ควรให้ความสนใจกับสุขภาพระบบย่อยอาหาร ควรรับประทานอาหารเป็นเวลา ดื่มน้ำให้เพียงพอ และเลือกอาหารที่สนับสนุนสุขภาพระบบย่อยอาหาร"
                }
            },
            'fate_line': {
                'strong': {
                    'en': "Traditional palmistry suggests excellent stamina and endurance. Consider consistent exercise routines and challenging physical activities.",
                    'th': "ตามหลักการดูลายมือแบบดั้งเดิม แสดงถึงความทนทานและความอดทนที่ยอดเยี่ยม ควรพิจารณาการออกกำลังกายอย่างสม่ำเสมอและกิจกรรมทางกายที่ท้าทาย"
                },
                'medium': {
                    'en': "Traditional palmistry suggests moderate stamina. Balance between activity and rest may be beneficial for optimal health.",
                    'th': "ตามหลักการดูลายมือแบบดั้งเดิม แสดงถึงความทนทานปานกลาง ความสมดุลระหว่างกิจกรรมและการพักผ่อนอาจเป็นประโยชน์สำหรับสุขภาพที่เหมาะสม"
                },
                'weak': {
                    'en': "Traditional palmistry suggests conserving energy. Consider gentle, regular exercise and adequate rest for building stamina gradually.",
                    'th': "ตามหลักการดูลายมือแบบดั้งเดิม แสดงถึงการอนุรักษ์พลังงาน ควรพิจารณาการออกกำลังกายเบาๆ เป็นประจำและการพักผ่อนที่เพียงพอเพื่อสร้างความทนทานอย่างค่อยเป็นค่อยไป"
                },
                'absent': {
                    'en': "Traditional palmistry suggests adaptable energy levels. Listen to your body's needs for activity and rest, and build routines gradually.",
                    'th': "ตามหลักการดูลายมือแบบดั้งเดิม แสดงถึงระดับพลังงานที่ปรับตัวได้ ควรฟังความต้องการของร่างกายสำหรับกิจกรรมและการพักผ่อน และสร้างกิจวัตรอย่างค่อยเป็นค่อยไป"
                }
            }
        }

        # Add finger interpretations from the previous version
        self.finger_interpretations = {
            'thumb': {
                'long': {
                    'en': "Strong willpower and determination. Traditional palmistry suggests good self-control and leadership potential.",
                    'th': "พลังใจและความมุ่งมั่นแข็งแกร่ง ตามหลักการดูลายมือแบบดั้งเดิม แสดงถึงการควบคุมตนเองที่ดีและศักยภาพความเป็นผู้นำ"
                },
                'medium': {
                    'en': "Balanced approach to goals. Traditional palmistry suggests good balance between determination and flexibility.",
                    'th': "แนวทางสมดุลต่อเป้าหมาย ตามหลักการดูลายมือแบบดั้งเดิม แสดงถึงความสมดุลที่ดีระหว่างความมุ่งมั่นและความยืดหยุ่น"
                },
                'short': {
                    'en': "Flexible and adaptable nature. Traditional palmistry suggests preference for cooperation over dominance.",
                    'th': "ธรรมชาติยืดหยุ่นและปรับตัวได้ ตามหลักการดูลายมือแบบดั้งเดิม แสดงถึงการชอบความร่วมมือมากกว่าการครอบงำ"
                }
            },
            'index': {
                'long': {
                    'en': "Natural leadership qualities. Traditional palmistry suggests confidence and ambition in achieving goals.",
                    'th': "คุณภาพผู้นำโดยธรรมชาติ ตามหลักการดูลายมือแบบดั้งเดิม แสดงถึงความมั่นใจและความทะเยอทะยานในการบรรลุเป้าหมาย"
                },
                'medium': {
                    'en': "Good balance of confidence and humility. Traditional palmistry suggests diplomatic leadership style.",
                    'th': "สมดุลดีระหว่างความมั่นใจและความถ่อมตน ตามหลักการดูลายมือแบบดั้งเดิม แสดงถึงรูปแบบการเป็นผู้นำที่ใช้การทูต"
                },
                'short': {
                    'en': "Prefers to follow rather than lead. Traditional palmistry suggests supportive and collaborative nature.",
                    'th': "ชอบเป็นผู้ตามมากกว่าเป็นผู้นำ ตามหลักการดูลายมือแบบดั้งเดิม แสดงถึงธรรมชาติที่สนับสนุนและร่วมมือ"
                }
            },
            'middle': {
                'long': {
                    'en': "Serious and responsible nature. Traditional palmistry suggests strong sense of duty and moral compass.",
                    'th': "ธรรมชาติจริงจังและมีความรับผิดชอบ ตามหลักการดูลายมือแบบดั้งเดิม แสดงถึงความรู้สึกหน้าที่และเข็มทิศศีลธรรมที่แข็งแกร่ง"
                },
                'medium': {
                    'en': "Well-balanced personality. Traditional palmistry suggests good judgment and practical wisdom.",
                    'th': "บุคลิกภาพที่สมดุลดี ตามหลักการดูลายมือแบบดั้งเดิม แสดงถึงการตัดสินใจที่ดีและปัญญาเชิงปฏิบัติ"
                },
                'short': {
                    'en': "Carefree and spontaneous. Traditional palmistry suggests preference for freedom and flexibility.",
                    'th': "ไร้กังวลและเป็นธรรมชาติ ตามหลักการดูลายมือแบบดั้งเดิม แสดงถึงการชอบอิสรภาพและความยืดหยุ่น"
                }
            },
            'ring': {
                'long': {
                    'en': "Creative and artistic tendencies. Traditional palmistry suggests strong aesthetic sense and emotional expression.",
                    'th': "แนวโน้มสร้างสรรค์และศิลปะ ตามหลักการดูลายมือแบบดั้งเดิม แสดงถึงความรู้สึกด้านความงามและการแสดงออกทางอารมณ์ที่แข็งแกร่ง"
                },
                'medium': {
                    'en': "Appreciates beauty and harmony. Traditional palmistry suggests balanced creative and practical abilities.",
                    'th': "ชื่นชมความงามและความกลมกลืน ตามหลักการดูลายมือแบบดั้งเดิม แสดงถึงความสามารถสร้างสรรค์และปฏิบัติที่สมดุล"
                },
                'short': {
                    'en': "Practical and straightforward. Traditional palmistry suggests preference for function over form.",
                    'th': "เป็นคนปฏิบัติและตรงไปตรงมา ตามหลักการดูลายมือแบบดั้งเดิม แสดงถึงการชอบการทำงานมากกว่ารูปแบบ"
                }
            },
            'pinky': {
                'long': {
                    'en': "Excellent communication skills. Traditional palmistry suggests persuasive abilities and social intelligence.",
                    'th': "ทักษะการสื่อสารยอดเยี่ยม ตามหลักการดูลายมือแบบดั้งเดิม แสดงถึงความสามารถในการโน้มน้าวและความฉลาดทางสังคม"
                },
                'medium': {
                    'en': "Good social abilities. Traditional palmistry suggests effective interpersonal communication.",
                    'th': "ความสามารถทางสังคมดี ตามหลักการดูลายมือแบบดั้งเดิม แสดงถึงการสื่อสารระหว่างบุคคลที่มีประสิทธิภาพ"
                },
                'short': {
                    'en': "Prefers actions over words. Traditional palmistry suggests direct and honest communication style.",
                    'th': "ชอบการกระทำมากกว่าคำพูด ตามหลักการดูลายมือแบบดั้งเดิม แสดงถึงรูปแบบการสื่อสารที่ตรงไปตรงมาและซื่อสัตย์"
                }
            }
        }

        # Add complete line and finger names
        self.palm_line_names = {
            'heart_line': {'en': 'Heart Line', 'th': 'เส้นหัวใจ'},
            'head_line': {'en': 'Head Line', 'th': 'เส้นสมอง'},
            'life_line': {'en': 'Life Line', 'th': 'เส้นชีวิต'},
            'mercury_line': {'en': 'Mercury Line (Health Line)', 'th': 'เส้นเมอร์คิวรี่ (เส้นสุขภาพ)'},
            'fate_line': {'en': 'Fate Line', 'th': 'เส้นชะตา'}
        }

        self.finger_names = {
            'thumb': {'en': 'Thumb', 'th': 'นิ้วหัวแม่มือ'},
            'index': {'en': 'Index Finger', 'th': 'นิ้วชี้'},
            'middle': {'en': 'Middle Finger', 'th': 'นิ้วกลาง'},
            'ring': {'en': 'Ring Finger', 'th': 'นิ้วนาง'},
            'pinky': {'en': 'Pinky Finger', 'th': 'นิ้วก้อย'}
        }

        # Add complete health aspects
        self.palm_health_aspects = {
            'heart_line': {'en': 'Cardiovascular & Emotional Health', 'th': 'สุขภาพหัวใจและอารมณ์'},
            'head_line': {'en': 'Mental Health & Cognitive Function', 'th': 'สุขภาพจิตและการทำงานของสมอง'},
            'life_line': {'en': 'Vitality & Physical Constitution', 'th': 'พลังชีวิตและร่างกาย'},
            'mercury_line': {'en': 'Digestive Health & Metabolism', 'th': 'สุขภาพระบบย่อยอาหารและการเผาผลาญ'},
            'fate_line': {'en': 'Stamina & Physical Endurance', 'th': 'ความทนทานและความอดทนทางกายภาพ'}
        }

        self.finger_aspects = {
            'thumb': {'en': 'Willpower & Self-Control', 'th': 'พลังใจและการควบคุมตนเอง'},
            'index': {'en': 'Leadership & Confidence', 'th': 'ความเป็นผู้นำและความมั่นใจ'},
            'middle': {'en': 'Responsibility & Moral Compass', 'th': 'ความรับผิดชอบและเข็มทิศศีลธรรม'},
            'ring': {'en': 'Creativity & Emotional Expression', 'th': 'ความคิดสร้างสรรค์และการแสดงออกทางอารมณ์'},
            'pinky': {'en': 'Communication & Social Intelligence', 'th': 'การสื่อสารและความฉลาดทางสังคม'}
        }
        
        # Face health interpretations based on traditional physiognomy
        self.face_health_interpretations = {
            'forehead': {
                'wide': {
                    'en': "Traditional face reading suggests good mental clarity and cognitive function. Consider brain-healthy activities like reading and puzzles.",
                    'th': "ตามหลักการดูหน้าแบบดั้งเดิม แสดงถึงความชัดเจนทางจิตใจและการทำงานของสมองที่ดี ควรพิจารณากิจกรรมที่ดีต่อสมอง เช่น การอ่านและปริศนา"
                },
                'medium': {
                    'en': "Traditional face reading suggests balanced mental approach. Regular mental exercises may help maintain cognitive health.",
                    'th': "ตามหลักการดูหน้าแบบดั้งเดิม แสดงถึงแนวทางจิตใจที่สมดุล การออกกำลังกายทางจิตใจเป็นประจำอาจช่วยรักษาสุขภาพทางปัญญา"
                },
                'narrow': {
                    'en': "Traditional face reading suggests focused thinking. Consider meditation and mindfulness practices for mental wellness.",
                    'th': "ตามหลักการดูหน้าแบบดั้งเดิม แสดงถึงการคิดที่มีสมาธิ ควรพิจารณาการทำสมาธิและการฝึกสติเพื่อสุขภาพจิตที่ดี"
                }
            },
            'eyes': {
                'bright': {
                    'en': "Traditional face reading suggests good vitality and energy. Maintain healthy sleep patterns and eye care.",
                    'th': "ตามหลักการดูหน้าแบบดั้งเดิม แสดงถึงพลังชีวิตและพลังงานที่ดี รักษารูปแบบการนอนหลับที่ดีและดูแลสุขภาพตา"
                },
                'medium': {
                    'en': "Traditional face reading suggests balanced energy levels. Consider adequate rest and eye protection from screens.",
                    'th': "ตามหลักการดูหน้าแบบดั้งเดิม แสดงถึงระดับพลังงานที่สมดุล ควรพักผ่อนให้เพียงพอและปกป้องตาจากหน้าจอ"
                },
                'tired': {
                    'en': "Traditional face reading suggests need for more rest. Consider improving sleep quality and reducing screen time.",
                    'th': "ตามหลักการดูหน้าแบบดั้งเดิม แสดงถึงความต้องการการพักผ่อนมากขึ้น ควรปรับปรุงคุณภาพการนอนและลดเวลาหน้าจอ"
                }
            },
            'nose': {
                'straight': {
                    'en': "Traditional face reading suggests good respiratory health. Maintain regular breathing exercises and fresh air exposure.",
                    'th': "ตามหลักการดูหน้าแบบดั้งเดิม แสดงถึงสุขภาพระบบหายใจที่ดี รักษาการออกกำลังกายการหายใจเป็นประจำและสัมผัสอากาศบริสุทธิ์"
                },
                'wide': {
                    'en': "Traditional face reading suggests strong constitution. Consider cardiovascular exercises for optimal health.",
                    'th': "ตามหลักการดูหน้าแบบดั้งเดิม แสดงถึงร่างกายที่แข็งแรง ควรพิจารณาการออกกำลังกายแบบคาร์ดิโอเพื่อสุขภาพที่เหมาะสม"
                },
                'narrow': {
                    'en': "Traditional face reading suggests sensitivity. Consider gentle breathing practices and avoiding irritants.",
                    'th': "ตามหลักการดูหน้าแบบดั้งเดิม แสดงถึงความอ่อนไหว ควรพิจารณาการฝึกหายใจเบาๆ และหลีกเลี่ยงสิ่งระคายเคือง"
                }
            },
            'mouth': {
                'full': {
                    'en': "Traditional face reading suggests good digestive health. Maintain balanced nutrition and mindful eating habits.",
                    'th': "ตามหลักการดูหน้าแบบดั้งเดิม แสดงถึงสุขภาพระบบย่อยอาหารที่ดี รักษาโภชนาการที่สมดุลและนิสัยการกินอย่างมีสติ"
                },
                'medium': {
                    'en': "Traditional face reading suggests balanced digestive function. Consider regular meal times and proper hydration.",
                    'th': "ตามหลักการดูหน้าแบบดั้งเดิม แสดงถึงการทำงานของระบบย่อยอาหารที่สมดุล ควรรับประทานอาหารเป็นเวลาและดื่มน้ำให้เพียงพอ"
                },
                'thin': {
                    'en': "Traditional face reading suggests careful eating habits. Consider nutrient-dense foods and digestive support.",
                    'th': "ตามหลักการดูหน้าแบบดั้งเดิม แสดงถึงนิสัยการกินที่ระมัดระวัง ควรพิจารณาอาหารที่มีสารอาหารหนาแน่นและสนับสนุนการย่อยอาหาร"
                }
            },
            'cheeks': {
                'full': {
                    'en': "Traditional face reading suggests good circulation and vitality. Maintain active lifestyle and proper nutrition.",
                    'th': "ตามหลักการดูหน้าแบบดั้งเดิม แสดงถึงการไหลเวียนและพลังชีวิตที่ดี รักษาวิถีชีวิตที่กระฉับกระเฉงและโภชนาการที่เหมาะสม"
                },
                'medium': {
                    'en': "Traditional face reading suggests balanced health. Consider regular exercise and stress management.",
                    'th': "ตามหลักการดูหน้าแบบดั้งเดิม แสดงถึงสุขภาพที่สมดุล ควรออกกำลังกายเป็นประจำและจัดการความเครียด"
                },
                'hollow': {
                    'en': "Traditional face reading suggests need for nourishment. Consider nutrient-rich foods and adequate rest.",
                    'th': "ตามหลักการดูหน้าแบบดั้งเดิม แสดงถึงความต้องการการบำรุง ควรพิจารณาอาหารที่มีสารอาหารสูงและการพักผ่อนที่เพียงพอ"
                }
            }
        }
        
        # Feature names in both languages
        self.face_feature_names = {
            'forehead': {'en': 'Forehead', 'th': 'หน้าผาก'},
            'eyes': {'en': 'Eyes', 'th': 'ดวงตา'},
            'nose': {'en': 'Nose', 'th': 'จมูก'},
            'mouth': {'en': 'Mouth', 'th': 'ปาก'},
            'cheeks': {'en': 'Cheeks', 'th': 'แก้ม'}
        }
        
        # Health aspects for face reading
        self.face_health_aspects = {
            'forehead': {'en': 'Mental Health & Cognitive Function', 'th': 'สุขภาพจิตและการทำงานของสมอง'},
            'eyes': {'en': 'Vitality & Energy Levels', 'th': 'พลังชีวิตและระดับพลังงาน'},
            'nose': {'en': 'Respiratory Health', 'th': 'สุขภาพระบบหายใจ'},
            'mouth': {'en': 'Digestive Health', 'th': 'สุขภาพระบบย่อยอาหาร'},
            'cheeks': {'en': 'Circulation & Overall Vitality', 'th': 'การไหลเวียนและพลังชีวิตโดยรวม'}
        }
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def get_length_category(self, length, thresholds):
        """Categorize length into long, medium, short"""
        if length > thresholds['long']:
            return 'long'
        elif length > thresholds['medium']:
            return 'medium'
        else:
            return 'short'
    
    def analyze_facial_features(self, landmarks):
        """Analyze facial features for health indicators"""
        face_analysis = {}
        
        # Convert landmarks to list for easier access
        landmark_points = [(lm.x, lm.y) for lm in landmarks.landmark]
        
        # Forehead analysis (based on width)
        forehead_left = landmark_points[21]  # Left eyebrow outer
        forehead_right = landmark_points[251]  # Right eyebrow outer
        forehead_width = abs(forehead_right[0] - forehead_left[0])
        
        if forehead_width > 0.15:
            forehead_category = 'wide'
        elif forehead_width > 0.12:
            forehead_category = 'medium'
        else:
            forehead_category = 'narrow'
        
        face_analysis['forehead'] = {
            'width': forehead_width,
            'category': forehead_category,
            'interpretation': self.face_health_interpretations['forehead'][forehead_category]
        }
        
        # Eyes analysis (based on openness and brightness - simplified)
        left_eye_top = landmark_points[159]
        left_eye_bottom = landmark_points[145]
        right_eye_top = landmark_points[386]
        right_eye_bottom = landmark_points[374]
        
        left_eye_height = abs(left_eye_top[1] - left_eye_bottom[1])
        right_eye_height = abs(right_eye_top[1] - right_eye_bottom[1])
        avg_eye_height = (left_eye_height + right_eye_height) / 2
        
        if avg_eye_height > 0.02:
            eyes_category = 'bright'
        elif avg_eye_height > 0.015:
            eyes_category = 'medium'
        else:
            eyes_category = 'tired'
        
        face_analysis['eyes'] = {
            'openness': avg_eye_height,
            'category': eyes_category,
            'interpretation': self.face_health_interpretations['eyes'][eyes_category]
        }
        
        # Nose analysis (based on width)
        nose_left = landmark_points[131]
        nose_right = landmark_points[360]
        nose_width = abs(nose_right[0] - nose_left[0])
        
        if nose_width > 0.04:
            nose_category = 'wide'
        elif nose_width > 0.03:
            nose_category = 'straight'
        else:
            nose_category = 'narrow'
        
        face_analysis['nose'] = {
            'width': nose_width,
            'category': nose_category,
            'interpretation': self.face_health_interpretations['nose'][nose_category]
        }
        
        # Mouth analysis (based on lip fullness)
        mouth_top = landmark_points[13]
        mouth_bottom = landmark_points[14]
        mouth_height = abs(mouth_top[1] - mouth_bottom[1])
        
        if mouth_height > 0.02:
            mouth_category = 'full'
        elif mouth_height > 0.015:
            mouth_category = 'medium'
        else:
            mouth_category = 'thin'
        
        face_analysis['mouth'] = {
            'fullness': mouth_height,
            'category': mouth_category,
            'interpretation': self.face_health_interpretations['mouth'][mouth_category]
        }
        
        # Cheeks analysis (based on fullness - simplified)
        left_cheek = landmark_points[116]
        right_cheek = landmark_points[345]
        face_center = landmark_points[1]
        
        left_cheek_distance = abs(left_cheek[0] - face_center[0])
        right_cheek_distance = abs(right_cheek[0] - face_center[0])
        avg_cheek_fullness = (left_cheek_distance + right_cheek_distance) / 2
        
        if avg_cheek_fullness > 0.08:
            cheeks_category = 'full'
        elif avg_cheek_fullness > 0.06:
            cheeks_category = 'medium'
        else:
            cheeks_category = 'hollow'
        
        face_analysis['cheeks'] = {
            'fullness': avg_cheek_fullness,
            'category': cheeks_category,
            'interpretation': self.face_health_interpretations['cheeks'][cheeks_category]
        }
        
        return face_analysis
    
    def generate_face_health_recommendations(self, face_analysis):
        """Generate health recommendations based on facial analysis"""
        recommendations = {
            'en': [
                "Maintain good sleep hygiene for overall facial health",
                "Stay hydrated to support skin and circulation",
                "Practice facial exercises to maintain muscle tone",
                "Protect your skin from sun damage with appropriate sunscreen",
                "Consider stress management techniques for overall wellness"
            ],
            'th': [
                "รักษาสุขอนามัยการนอนที่ดีเพื่อสุขภาพใบหน้าโดยรวม",
                "ดื่มน้ำให้เพียงพอเพื่อสนับสนุนผิวหนังและการไหลเวียน",
                "ฝึกการออกกำลังกายใบหน้าเพื่อรักษาความกระชับของกล้ามเนื้อ",
                "ปกป้องผิวจากแสงแดดด้วยครีมกันแดดที่เหมาะสม",
                "พิจารณาเทคนิคการจัดการความเครียดเพื่อสุขภาพโดยรวม"
            ]
        }
        
        # Add specific recommendations based on analysis
        if face_analysis['eyes']['category'] == 'tired':
            recommendations['en'].append("Consider improving sleep quality and reducing screen time")
            recommendations['th'].append("พิจารณาปรับปรุงคุณภาพการนอนและลดเวลาหน้าจอ")
        
        if face_analysis['cheeks']['category'] == 'hollow':
            recommendations['en'].append("Focus on nutrient-rich foods and adequate caloric intake")
            recommendations['th'].append("เน้นอาหารที่มีสารอาหารสูงและการรับพลังงานที่เพียงพอ")
        
        if face_analysis['forehead']['category'] == 'narrow':
            recommendations['en'].append("Practice meditation and mindfulness for mental clarity")
            recommendations['th'].append("ฝึกการทำสมาธิและการมีสติเพื่อความชัดเจนทางจิตใจ")
        
        return recommendations

    def analyze_palm_lines(self, landmarks):
        """Analyze major palm lines with complete health interpretations"""
        analysis = {}
        
        # Heart line (from pinky side to index finger area)
        heart_line_start = landmarks[17]  # Pinky base
        heart_line_end = landmarks[5]     # Index finger base
        heart_line_length = self.calculate_distance(heart_line_start, heart_line_end)
        
        # Head line (across the palm)
        head_line_start = landmarks[17]   # Pinky side
        head_line_end = landmarks[4]      # Thumb tip area
        head_line_length = self.calculate_distance(head_line_start, head_line_end)
        
        # Life line (around thumb)
        life_line_start = landmarks[2]    # Thumb joint
        life_line_end = landmarks[0]      # Wrist
        life_line_length = self.calculate_distance(life_line_start, life_line_end)
        
        # Mercury line (health line) - check if present
        mercury_present = self.check_line_presence(landmarks, 0, 17)
        mercury_category = 'present' if mercury_present else 'absent'
        
        # Fate line - check strength
        fate_line_start = landmarks[0]    # Wrist
        fate_line_end = landmarks[9]      # Middle finger base
        fate_line_length = self.calculate_distance(fate_line_start, fate_line_end)
        
        # Categorize lengths
        heart_category = self.get_length_category(heart_line_length, {'long': 0.3, 'medium': 0.2})
        head_category = self.get_length_category(head_line_length, {'long': 0.4, 'medium': 0.3})
        life_category = self.get_length_category(life_line_length, {'long': 0.3, 'medium': 0.2})
        
        if fate_line_length > 0.35:
            fate_category = 'strong'
        elif fate_line_length > 0.25:
            fate_category = 'medium'
        elif fate_line_length > 0.15:
            fate_category = 'weak'
        else:
            fate_category = 'absent'
        
        analysis['heart_line'] = {
            'length': heart_line_length,
            'category': heart_category,
            'interpretation': self.palm_health_interpretations['heart_line'][heart_category]
        }
        
        analysis['head_line'] = {
            'length': head_line_length,
            'category': head_category,
            'interpretation': self.palm_health_interpretations['head_line'][head_category]
        }
        
        analysis['life_line'] = {
            'length': life_line_length,
            'category': life_category,
            'interpretation': self.palm_health_interpretations['life_line'][life_category]
        }
        
        analysis['mercury_line'] = {
            'present': mercury_present,
            'category': mercury_category,
            'interpretation': self.palm_health_interpretations['mercury_line'][mercury_category]
        }
        
        analysis['fate_line'] = {
            'length': fate_line_length,
            'category': fate_category,
            'interpretation': self.palm_health_interpretations['fate_line'][fate_category]
        }
        
        return analysis

    def analyze_fingers(self, landmarks):
        """Analyze finger characteristics with complete interpretations"""
        finger_analysis = {}
        
        # Finger tips and bases for length calculation
        fingers = {
            'thumb': (landmarks[4], landmarks[2]),
            'index': (landmarks[8], landmarks[5]),
            'middle': (landmarks[12], landmarks[9]),
            'ring': (landmarks[16], landmarks[13]),
            'pinky': (landmarks[20], landmarks[17])
        }
        
        for finger_name, (tip, base) in fingers.items():
            length = self.calculate_distance(tip, base)
            category = self.get_length_category(length, {'long': 0.15, 'medium': 0.1})
            
            finger_analysis[finger_name] = {
                'length': length,
                'category': category,
                'interpretation': self.finger_interpretations[finger_name][category]
            }
        
        return finger_analysis

    def check_line_presence(self, landmarks, start_idx, end_idx, threshold=0.02):
        """Check if a line is present based on distance analysis"""
        start_point = landmarks[start_idx]
        end_point = landmarks[end_idx]
        distance = self.calculate_distance(start_point, end_point)
        
        # Simplified presence detection - in real app, would analyze actual palm creases
        import random
        presence_factor = distance * random.uniform(0.8, 1.2)
        
        return presence_factor > threshold

    def generate_palm_health_recommendations(self, palm_analysis, finger_analysis):
        """Generate comprehensive health recommendations based on palm analysis"""
        recommendations = {
            'en': [
                "Remember to stay hydrated throughout the day for overall health",
                "Consider incorporating mindfulness practices into your routine",
                "Regular physical activity appropriate for your energy levels is beneficial",
                "Balanced nutrition supports overall wellbeing and vitality",
                "Adequate sleep is essential for health maintenance and recovery"
            ],
            'th': [
                "อย่าลืมดื่มน้ำให้เพียงพอตลอดทั้งวันเพื่อสุขภาพโดยรวม",
                "พิจารณาการฝึกสติในชีวิตประจำวัน",
                "การออกกำลังกายเป็นประจำที่เหมาะสมกับระดับพลังงานของคุณเป็นประโยชน์",
                "โภชนาการที่สมดุลช่วยสนับสนุนสุขภาพและพลังชีวิตโดยรวม",
                "การนอนหลับที่เพียงพอเป็นสิ่งสำคัญสำหรับการรักษาสุขภาพและการฟื้นฟู"
            ]
        }
        
        # Add specific recommendations based on palm analysis
        if palm_analysis['life_line']['category'] == 'short':
            recommendations['en'].append("Consider energy conservation techniques and gentle exercise like yoga or tai chi")
            recommendations['th'].append("พิจารณาเทคนิคการอนุรักษ์พลังงานและการออกกำลังกายเบาๆ เช่น โยคะหรือไทชิ")
        
        if palm_analysis['head_line']['category'] == 'long':
            recommendations['en'].append("Mental exercises like puzzles and reading may help maintain cognitive function")
            recommendations['th'].append("การออกกำลังกายทางจิตใจ เช่น ปริศนาและการอ่าน อาจช่วยรักษาการทำงานของสมอง")
        
        if palm_analysis['heart_line']['category'] == 'short':
            recommendations['en'].append("Stress management practices may support emotional and cardiovascular wellbeing")
            recommendations['th'].append("การฝึกจัดการความเครียดอาจช่วยสนับสนุนสุขภาพทางอารมณ์และหัวใจ")
        
        if palm_analysis['mercury_line']['category'] == 'absent':
            recommendations['en'].append("Pay attention to digestive health with regular meals and fiber-rich foods")
            recommendations['th'].append("ให้ความสนใจกับสุขภาพระบบย่อยอาหารด้วยการรับประทานอาหารเป็นเวลาและอาหารที่มีเส้นใยสูง")
        
        if palm_analysis['fate_line']['category'] in ['weak', 'absent']:
            recommendations['en'].append("Build stamina gradually with consistent, gentle exercise routines")
            recommendations['th'].append("สร้างความทนทานอย่างค่อยเป็นค่อยไปด้วยการออกกำลังกายเบาๆ อย่างสม่ำเสมอ")
        
        # Add recommendations based on finger analysis
        if finger_analysis['thumb']['category'] == 'short':
            recommendations['en'].append("Practice self-discipline techniques to strengthen willpower")
            recommendations['th'].append("ฝึกเทคนิคการมีวินัยในตนเองเพื่อเสริมสร้างพลังใจ")
        
        if finger_analysis['index']['category'] == 'long':
            recommendations['en'].append("Channel your leadership qualities into positive health habits")
            recommendations['th'].append("นำคุณภาพความเป็นผู้นำของคุณไปใช้ในการสร้างนิสัยสุขภาพที่ดี")
        
        return recommendations

def display_bilingual_result(title_en, title_th, content_en, content_th):
    """Display content in both languages side by side"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**🇺🇸 {title_en}:**")
        st.write(f"• {content_en}")
    
    with col2:
        st.markdown(f"**🇹🇭 {title_th}:**")
        st.write(f"• {content_th}")
    
    st.write("")

def palm_analysis_tab(analyzer):
    """Complete palm analysis functionality"""
    st.header("🖐️ Palm Health Analysis / การวิเคราะห์สุขภาพจากลายมือ")
    
    # Important disclaimer
    st.error("""
    **🇺🇸 IMPORTANT DISCLAIMER:** This application is for entertainment purposes only. 
    Palm reading is not a substitute for professional medical advice, diagnosis, or treatment.
    Always seek the advice of your physician or other qualified health provider.
    
    **🇹🇭 คำเตือนสำคัญ:** แอปพลิเคชันนี้มีไว้เพื่อความบันเทิงเท่านั้น 
    การดูลายมือไม่ใช่การทดแทนคำแนะนำทางการแพทย์ การวินิจฉัย หรือการรักษา
    โปรดปรึกษาแพทย์หรือผู้ให้บริการด้านสุขภาพที่มีคุณสมบัติเหมาะสมเสมอ
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Capture Your Palm / ถ่ายภาพฝ่ามือ")
        
        st.info("""
        **🇺🇸 Tips for best results:**
        - Position your hand clearly in front of the camera
        - Ensure good lighting for best results
        - Keep your palm open and fingers spread
        - Hold steady for a clear photo
        
        **🇹🇭 เคล็ดลับสำหรับผลลัพธ์ที่ดีที่สุด:**
        - วางมือให้ชัดเจนหน้ากล้อง
        - ให้แสงสว่างเพียงพอเพื่อผลลัพธ์ที่ดีที่สุด
        - เปิดฝ่ามือและกางนิ้ว
        - จับให้นิ่งเพื่อภาพที่ชัดเจน
        """)
        
        palm_camera = st.camera_input("Take a photo of your palm / ถ่ายภาพฝ่ามือของคุณ", key="palm_camera")
        
        if palm_camera is not None:
            image = Image.open(palm_camera)
            image_array = np.array(image)
            
            results = analyzer.hands.process(image_array)
            
            if results.multi_hand_landmarks:
                annotated_image = image_array.copy()
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated_image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                
                st.image(annotated_image, caption="Detected Hand Landmarks / จุดสำคัญของมือที่ตรวจพบ")
                
                with col2:
                    st.subheader("🔮 Palm Analysis Results / ผลการวิเคราะห์ลายมือ")
                    
                    # Analyze the first detected hand
                    hand_landmarks = results.multi_hand_landmarks[0]
                    
                    # Palm lines analysis
                    palm_analysis = analyzer.analyze_palm_lines(hand_landmarks.landmark)
                    
                    # Finger analysis
                    finger_analysis = analyzer.analyze_fingers(hand_landmarks.landmark)
                    
                    st.info("""
                    **🇺🇸 Reminder:** Based on traditional palmistry principles for entertainment only.
                    
                    **🇹🇭 การเตือน:** อิงตามหลักการดูลายมือแบบดั้งเดิมเพื่อความบันเทิงเท่านั้น
                    """)
                    
                    st.subheader("📏 Major Palm Lines / เส้นหลักในฝ่ามือ")
                    
                    for line_key, line_data in palm_analysis.items():
                        line_name_en = analyzer.palm_line_names[line_key]['en']
                        line_name_th = analyzer.palm_line_names[line_key]['th']
                        health_aspect_en = analyzer.palm_health_aspects[line_key]['en']
                        health_aspect_th = analyzer.palm_health_aspects[line_key]['th']
                        interpretation_en = line_data['interpretation']['en']
                        interpretation_th = line_data['interpretation']['th']
                        
                        st.markdown(f"### {line_name_en} / {line_name_th}")
                        st.markdown(f"**{health_aspect_en} / {health_aspect_th}**")
                        
                        display_bilingual_result(
                            "Traditional Interpretation", "การตีความแบบดั้งเดิม",
                            interpretation_en, interpretation_th
                        )
                    
                    st.subheader("👆 Finger Analysis / การวิเคราะห์นิ้ว")
                    
                    for finger_key, finger_data in finger_analysis.items():
                        finger_name_en = analyzer.finger_names[finger_key]['en']
                        finger_name_th = analyzer.finger_names[finger_key]['th']
                        finger_aspect_en = analyzer.finger_aspects[finger_key]['en']
                        finger_aspect_th = analyzer.finger_aspects[finger_key]['th']
                        interpretation_en = finger_data['interpretation']['en']
                        interpretation_th = finger_data['interpretation']['th']
                        
                        st.markdown(f"### {finger_name_en} / {finger_name_th}")
                        st.markdown(f"**{finger_aspect_en} / {finger_aspect_th}**")
                        
                        display_bilingual_result(
                            "Traditional Interpretation", "การตีความแบบดั้งเดิม",
                            interpretation_en, interpretation_th
                        )
                    
                    # Overall personality summary
                    st.subheader("🌟 Personality Summary / สรุปบุคลิกภาพ")
                    
                    col_en, col_th = st.columns(2)
                    
                    with col_en:
                        st.info("""
                        **🇺🇸 English:**
                        
                        Based on your palm analysis, you show a unique combination of traits. 
                        Remember that palm reading is for entertainment purposes and should not 
                        be used for making important life decisions. Your future is in your hands!
                        """)
                    
                    with col_th:
                        st.info("""
                        **🇹🇭 ไทย:**
                        
                        จากการวิเคราะห์ลายมือของคุณ แสดงให้เห็นการผสมผสานของลักษณะที่เป็นเอกลักษณ์ 
                        โปรดจำไว้ว่าการดูลายมือเป็นเพียงความบันเทิงและไม่ควรใช้สำหรับการตัดสินใจสำคัญในชีวิต 
                        อนาคตของคุณอยู่ในมือของคุณเอง!
                        """)
                    
                    # Health recommendations
                    st.subheader("💡 Health & Wellness Suggestions / คำแนะนำสุขภาพและความเป็นอยู่ที่ดี")
                    
                    recommendations = analyzer.generate_palm_health_recommendations(palm_analysis, finger_analysis)
                    
                    col_en, col_th = st.columns(2)
                    
                    with col_en:
                        st.markdown("**🇺🇸 English:**")
                        for rec in recommendations['en']:
                            st.markdown(f"• {rec}")
                    
                    with col_th:
                        st.markdown("**🇹🇭 ไทย:**")
                        for rec in recommendations['th']:
                            st.markdown(f"• {rec}")
                    
                    # Download palm analysis report
                    st.subheader("📄 Download Report / ดาวน์โหลดรายงาน")
                    
                    # Create comprehensive bilingual report
                    report_content = f"""
COMPREHENSIVE PALM ANALYSIS REPORT / รายงานการวิเคราะห์ลายมือแบบครอบคลุม
===============================================================================

IMPORTANT DISCLAIMER / คำเตือนสำคัญ:
🇺🇸 This analysis is based on traditional palmistry beliefs and is for entertainment purposes only.
   It is NOT a substitute for professional medical advice, diagnosis, or treatment.
   
🇹🇭 การวิเคราะห์นี้อิงตามความเชื่อการดูลายมือแบบดั้งเดิมและมีไว้เพื่อความบันเทิงเท่านั้น
   ไม่ใช่การทดแทนคำแนะนำทางการแพทย์ การวินิจฉัย หรือการรักษาจากผู้เชี่ยวชาญ

===============================================================================

MAJOR PALM LINES ANALYSIS / การวิเคราะห์เส้นหลักในฝ่ามือ:
"""
                    
                    for line_key, line_data in palm_analysis.items():
                        line_name_en = analyzer.palm_line_names[line_key]['en']
                        line_name_th = analyzer.palm_line_names[line_key]['th']
                        health_aspect_en = analyzer.palm_health_aspects[line_key]['en']
                        health_aspect_th = analyzer.palm_health_aspects[line_key]['th']
                        interpretation_en = line_data['interpretation']['en']
                        interpretation_th = line_data['interpretation']['th']
                        
                        report_content += f"""
{line_name_en} / {line_name_th} - {health_aspect_en} / {health_aspect_th}:
🇺🇸 {interpretation_en}
🇹🇭 {interpretation_th}

"""
                    
                    report_content += f"""
FINGER ANALYSIS / การวิเคราะห์นิ้ว:
"""
                    
                    for finger_key, finger_data in finger_analysis.items():
                        finger_name_en = analyzer.finger_names[finger_key]['en']
                        finger_name_th = analyzer.finger_names[finger_key]['th']
                        finger_aspect_en = analyzer.finger_aspects[finger_key]['en']
                        finger_aspect_th = analyzer.finger_aspects[finger_key]['th']
                        interpretation_en = finger_data['interpretation']['en']
                        interpretation_th = finger_data['interpretation']['th']
                        
                        report_content += f"""
{finger_name_en} / {finger_name_th} - {finger_aspect_en} / {finger_aspect_th}:
🇺🇸 {interpretation_en}
🇹🇭 {interpretation_th}

"""
                    
                    report_content += f"""
HEALTH & WELLNESS RECOMMENDATIONS / คำแนะนำสุขภาพและความเป็นอยู่ที่ดี:
"""
                    
                    for i in range(len(recommendations['en'])):
                        report_content += f"""
🇺🇸 {recommendations['en'][i]}
🇹🇭 {recommendations['th'][i]}
"""
                    
                    report_content += """
===============================================================================

FINAL REMINDER / การเตือนสุดท้าย:
🇺🇸 Always consult with healthcare professionals for any health concerns.
🇹🇭 ปรึกษาผู้เชี่ยวชาญด้านสุขภาพเสมอสำหรับความกังวลด้านสุขภาพใดๆ
"""
                    
                    st.download_button(
                        label="📄 Download Complete Palm Analysis Report / ดาวน์โหลดรายงานการวิเคราะห์ลายมือแบบครอบคลุม",
                        data=report_content,
                        file_name="complete_palm_analysis_report.txt",
                        mime="text/plain",
                        key="palm_report_download"
                    )
            
            else:
                col_warn1, col_warn2 = st.columns(2)
                
                with col_warn1:
                    st.warning("⚠️ **English**: No hand detected in the image. Please try again with better lighting and hand positioning.")
                
                with col_warn2:
                    st.warning("⚠️ **ไทย**: ไม่พบมือในภาพ กรุณาลองใหม่ด้วยแสงที่ดีกว่าและการวางตำแหน่งมือที่ชัดเจน")

def face_analysis_tab(analyzer):
    """Face analysis functionality"""
    st.header("👤 Face Health Analysis / การวิเคราะห์สุขภาพจากใบหน้า")
    
    # Important disclaimer
    st.error("""
    **🇺🇸 IMPORTANT DISCLAIMER:** This application is for entertainment purposes only. 
    Face reading is not a substitute for professional medical advice, diagnosis, or treatment.
    
    **🇹🇭 คำเตือนสำคัญ:** แอปพลิเคชันนี้มีไว้เพื่อความบันเทิงเท่านั้น 
    การดูหน้าไม่ใช่การทดแทนคำแนะนำทางการแพทย์ การวินิจฉัย หรือการรักษา
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Capture Your Face / ถ่ายภาพใบหน้า")
        
        st.info("""
        **🇺🇸 Tips for best results:**
        - Face the camera directly
        - Ensure good lighting
        - Keep a neutral expression
        - Remove glasses if possible
        
        **🇹🇭 เคล็ดลับสำหรับผลลัพธ์ที่ดีที่สุด:**
        - หันหน้าเข้าหากล้องโดยตรง
        - ให้แสงสว่างเพียงพอ
        - ทำหน้าเป็นกลาง
        - ถอดแว่นตาหากเป็นไปได้
        """)
        
        face_camera = st.camera_input("Take a photo of your face / ถ่ายภาพใบหน้าของคุณ", key="face_camera")
        
        if face_camera is not None:
            image = Image.open(face_camera)
            image_array = np.array(image)
            
            # Convert BGR to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            results = analyzer.face_mesh.process(rgb_image)
            
            if results.multi_face_landmarks:
                annotated_image = image_array.copy()
                
                for face_landmarks in results.multi_face_landmarks:
                    # Draw face mesh
                    mp_drawing.draw_landmarks(
                        annotated_image,
                        face_landmarks,
                        mp_face_mesh.FACEMESH_CONTOURS,
                        None,
                        mp_drawing_styles.get_default_face_mesh_contours_style()
                    )
                
                st.image(annotated_image, caption="Detected Face Landmarks / จุดสำคัญของใบหน้าที่ตรวจพบ")
                
                with col2:
                    st.subheader("🔮 Face Analysis Results / ผลการวิเคราะห์ใบหน้า")
                    
                    # Analyze facial features
                    face_landmarks = results.multi_face_landmarks[0]
                    face_analysis = analyzer.analyze_facial_features(face_landmarks)
                    
                    st.info("""
                    **🇺🇸 Reminder:** Based on traditional face reading principles for entertainment only.
                    
                    **🇹🇭 การเตือน:** อิงตามหลักการดูหน้าแบบดั้งเดิมเพื่อความบันเทิงเท่านั้น
                    """)
                    
                    st.subheader("📊 Facial Health Indicators / ตัวบ่งชี้สุขภาพจากใบหน้า")
                    
                    for feature_key, feature_data in face_analysis.items():
                        feature_name_en = analyzer.face_feature_names[feature_key]['en']
                        feature_name_th = analyzer.face_feature_names[feature_key]['th']
                        health_aspect_en = analyzer.face_health_aspects[feature_key]['en']
                        health_aspect_th = analyzer.face_health_aspects[feature_key]['th']
                        interpretation_en = feature_data['interpretation']['en']
                        interpretation_th = feature_data['interpretation']['th']
                        
                        st.markdown(f"### {feature_name_en} / {feature_name_th}")
                        st.markdown(f"**{health_aspect_en} / {health_aspect_th}**")
                        
                        display_bilingual_result(
                            "Traditional Interpretation", "การตีความแบบดั้งเดิม",
                            interpretation_en, interpretation_th
                        )
                    
                    # Face health recommendations
                    st.subheader("💡 Wellness Suggestions / คำแนะนำสุขภาพ")
                    
                    recommendations = analyzer.generate_face_health_recommendations(face_analysis)
                    
                    col_en, col_th = st.columns(2)
                    
                    with col_en:
                        st.markdown("**🇺🇸 English:**")
                        for rec in recommendations['en']:
                            st.markdown(f"• {rec}")
                    
                    with col_th:
                        st.markdown("**🇹🇭 ไทย:**")
                        for rec in recommendations['th']:
                            st.markdown(f"• {rec}")
                    
                    # Download face analysis report
                    st.subheader("📄 Download Report / ดาวน์โหลดรายงาน")
                    
                    # Create bilingual report
                    report_content = f"""
FACE HEALTH ANALYSIS REPORT / รายงานการวิเคราะห์สุขภาพจากใบหน้า
===============================================================================

IMPORTANT DISCLAIMER / คำเตือนสำคัญ:
🇺🇸 This analysis is based on traditional face reading principles and is for entertainment purposes only.
   It is NOT a substitute for professional medical advice, diagnosis, or treatment.
   
🇹🇭 การวิเคราะห์นี้อิงตามหลักการดูหน้าแบบดั้งเดิมและมีไว้เพื่อความบันเทิงเท่านั้น
   ไม่ใช่การทดแทนคำแนะนำทางการแพทย์ การวินิจฉัย หรือการรักษาจากผู้เชี่ยวชาญ

===============================================================================

FACIAL HEALTH INDICATORS / ตัวบ่งชี้สุขภาพจากใบหน้า:
"""
                    
                    for feature_key, feature_data in face_analysis.items():
                        feature_name_en = analyzer.face_feature_names[feature_key]['en']
                        feature_name_th = analyzer.face_feature_names[feature_key]['th']
                        health_aspect_en = analyzer.face_health_aspects[feature_key]['en']
                        health_aspect_th = analyzer.face_health_aspects[feature_key]['th']
                        interpretation_en = feature_data['interpretation']['en']
                        interpretation_th = feature_data['interpretation']['th']
                        
                        report_content += f"""
{feature_name_en} / {feature_name_th} - {health_aspect_en} / {health_aspect_th}:
🇺🇸 {interpretation_en}
🇹🇭 {interpretation_th}

"""
                    
                    report_content += f"""
WELLNESS SUGGESTIONS / คำแนะนำสุขภาพ:
"""
                    
                    for i in range(len(recommendations['en'])):
                        report_content += f"""
🇺🇸 {recommendations['en'][i]}
🇹🇭 {recommendations['th'][i]}
"""
                    
                    report_content += """
===============================================================================

FINAL REMINDER / การเตือนสุดท้าย:
🇺🇸 Always consult with healthcare professionals for any health concerns.
🇹🇭 ปรึกษาผู้เชี่ยวชาญด้านสุขภาพเสมอสำหรับความกังวลด้านสุขภาพใดๆ
"""
                    
                    st.download_button(
                        label="📄 Download Face Analysis Report / ดาวน์โหลดรายงานการวิเคราะห์ใบหน้า",
                        data=report_content,
                        file_name="face_health_analysis_report.txt",
                        mime="text/plain",
                        key="face_report_download"
                    )
            
            else:
                st.warning("⚠️ **English**: No face detected in the image. Please try again with better lighting and positioning.")
                st.warning("⚠️ **ไทย**: ไม่พบใบหน้าในภาพ กรุณาลองใหม่ด้วยแสงที่ดีกว่าและการวางตำแหน่งที่ชัดเจน")

def main():
    st.set_page_config(
        page_title="Combined Health Analysis / การวิเคราะห์สุขภาพรวม",
        page_icon="🔮",
        layout="wide"
    )
    
    # Header
    st.title("🔮 Combined Health Analysis App")
    st.title("แอปพลิเคชันการวิเคราะห์สุขภาพรวม")
    
    st.markdown("""
    **🇺🇸 English**: AI-powered health analysis through palm reading and face reading  
    **🇹🇭 ไทย**: การวิเคราะห์สุขภาพด้วย AI ผ่านการดูลายมือและการดูหน้า
    """)
    
    # Initialize analyzer
    analyzer = CombinedHealthAnalyzer()
    
    # Create tabs
    tab1, tab2 = st.tabs([
        "🖐️ Palm Analysis / การดูลายมือ", 
        "👤 Face Analysis / การดูหน้า"
    ])
    
    with tab1:
        palm_analysis_tab(analyzer)
    
    with tab2:
        face_analysis_tab(analyzer)
    
    # Sidebar with general information
    with st.sidebar:
        st.header("📋 General Information / ข้อมูลทั่วไป")
        
        st.subheader("🇺🇸 About This App")
        st.markdown("""
        This application combines traditional palm reading and face reading 
        principles with modern AI technology for entertainment purposes.
        
        **Features:**
        - Palm health analysis
        - Facial feature health analysis
        - Bilingual support (Thai/English)
        - Downloadable reports
        - Camera integration
        """)
        
        st.subheader("🇹🇭 เกี่ยวกับแอปนี้")
        st.markdown("""
        แอปพลิเคชันนี้รวมหลักการดูลายมือและการดูหน้าแบบดั้งเดิม
        เข้ากับเทคโนโลยี AI สมัยใหม่เพื่อความบันเทิง
        
        **คุณสมบัติ:**
        - การวิเคราะห์สุขภาพจากลายมือ
        - การวิเคราะห์สุขภาพจากใบหน้า
        - รองรับสองภาษา (ไทย/อังกฤษ)
        - รายงานที่ดาวน์โหลดได้
        - การเชื่อมต่อกล้อง
        """)
        
        st.markdown("---")
        
        st.subheader("⚠️ Important Disclaimers / คำเตือนสำคัญ")
        st.error("""
        **🇺🇸 English:**
        - For entertainment purposes only
        - Not medical advice
        - Consult healthcare professionals for health concerns
        
        **🇹🇭 ไทย:**
        - เพื่อความบันเทิงเท่านั้น
        - ไม่ใช่คำแนะนำทางการแพทย์
        - ปรึกษาผู้เชี่ยวชาญด้านสุขภาพสำหรับปัญหาสุขภาพ
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p><em>🔮 Combined Health Analysis App - For Entertainment Purposes Only 🔮</em></p>
        <p><em>🔮 แอปการวิเคราะห์สุขภาพรวม - เพื่อความบันเทิงเท่านั้น 🔮</em></p>
        <p><small>
            🇺🇸 This application uses AI and computer vision for traditional health analysis. 
            Results should not be used for making health decisions.<br>
            🇹🇭 แอปพลิเคชันนี้ใช้ AI และคอมพิวเตอร์วิชันสำหรับการวิเคราะห์สุขภาพแบบดั้งเดิม 
            ไม่ควรใช้ผลลัพธ์สำหรับการตัดสินใจด้านสุขภาพ
        </small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
