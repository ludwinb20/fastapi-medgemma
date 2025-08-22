import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import logging
from typing import Tuple, Dict, Any
import os

logger = logging.getLogger(__name__)

class MedicalClassifier:
    """Clasificador para determinar si una pregunta es médica o no"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.medical_keywords = [
            # Síntomas comunes
            "dolor", "fiebre", "tos", "dolor de cabeza", "náuseas", "vómitos", "diarrea", "estreñimiento",
            "fatiga", "mareos", "dolor de estómago", "dolor de espalda", "dolor de garganta",
            "síntomas", "malestar", "inflamación", "hinchazón", "picazón", "erupción",
            
            # Enfermedades
            "enfermedad", "infección", "virus", "bacteria", "cáncer", "diabetes", "hipertensión",
            "asma", "artritis", "alergia", "gripe", "resfriado", "neumonía", "bronquitis",
            "gastritis", "úlcera", "hepatitis", "migraña", "epilepsia", "depresión", "ansiedad",
            
            # Tratamientos
            "medicamento", "antibiótico", "analgésico", "tratamiento", "terapia", "cirugía",
            "operación", "medicina", "pastilla", "inyección", "vacuna", "remedio", "cura",
            
            # Anatomía y fisiología
            "corazón", "pulmón", "hígado", "riñón", "cerebro", "sangre", "hueso", "músculo",
            "piel", "ojo", "oído", "nariz", "boca", "estómago", "intestino", "vejiga",
            "útero", "ovario", "testículo", "pene", "vagina", "mama", "pecho",
            
            # Procedimientos médicos
            "examen", "análisis", "radiografía", "resonancia", "tomografía", "ecografía",
            "endoscopia", "biopsia", "análisis de sangre", "prueba", "diagnóstico",
            
            # Profesionales de salud
            "médico", "doctor", "enfermero", "especialista", "cirujano", "pediatra",
            "ginecólogo", "cardiólogo", "dermatólogo", "psiquiatra", "farmacéutico",
            
            # Términos médicos
            "paciente", "consulta", "historia clínica", "receta", "prescripción",
            "dosis", "efectos secundarios", "contraindicaciones", "prevención",
            "pronóstico", "complicaciones", "rehabilitación", "recuperación",
            
            # Emergencias
            "emergencia", "urgencia", "accidente", "trauma", "fractura", "hemorragia",
            "desmayo", "convulsión", "paro cardíaco", "shock", "intoxicación",
            
            # Salud general
            "salud", "bienestar", "nutrición", "ejercicio", "dieta", "peso",
            "presión arterial", "temperatura", "pulso", "respiración", "oxigenación",
            
            # Edades y grupos
            "bebé", "niño", "adolescente", "adulto", "mayor", "embarazo", "parto",
            "menopausia", "pubertad", "vejez", "geriatría", "pediatría",
            
            # Especialidades médicas
            "cardiología", "dermatología", "ginecología", "pediatría", "psiquiatría",
            "neurología", "ortopedia", "oftalmología", "otorrinolaringología",
            "urología", "gastroenterología", "endocrinología", "reumatología"
        ]
        
        self.non_medical_keywords = [
            # Política
            "política", "gobierno", "presidente", "ministro", "congreso", "senado",
            "elecciones", "votación", "partido político", "democracia", "dictadura",
            "revolución", "protesta", "manifestación", "huelga", "sindicato",
            
            # Economía
            "economía", "dinero", "precio", "inflación", "desempleo", "bolsa",
            "mercado", "inversión", "banco", "crédito", "deuda", "impuestos",
            "comercio", "exportación", "importación", "PIB", "recesión",
            
            # Deportes
            "fútbol", "béisbol", "baloncesto", "tenis", "golf", "natación",
            "atletismo", "ciclismo", "boxeo", "lucha", "esquí", "surf",
            "equipo", "jugador", "entrenador", "campeonato", "liga", "torneo",
            
            # Entretenimiento
            "película", "serie", "televisión", "música", "cantante", "actor",
            "director", "libro", "novela", "poesía", "arte", "pintura",
            "teatro", "concierto", "festival", "videojuego", "juego",
            
            # Historia
            "historia", "guerra", "batalla", "revolución", "independencia",
            "colonización", "imperio", "rey", "reina", "emperador", "dictador",
            "siglo", "año", "fecha", "acontecimiento", "evento histórico",
            
            # Geografía
            "país", "ciudad", "estado", "provincia", "región", "continente",
            "océano", "mar", "río", "montaña", "desierto", "isla", "capital",
            "frontera", "clima", "temperatura", "lluvia", "nieve",
            
            # Tecnología
            "computadora", "teléfono", "internet", "redes sociales", "software",
            "hardware", "programación", "aplicación", "sitio web", "correo",
            "video", "foto", "cámara", "televisor", "radio", "satélite",
            
            # Educación
            "escuela", "universidad", "colegio", "profesor", "estudiante",
            "clase", "curso", "examen", "tarea", "investigación", "tesis",
            "grado", "maestría", "doctorado", "carrera", "materia",
            
            # Religión
            "religión", "dios", "iglesia", "templo", "mezquita", "sinagoga",
            "sacerdote", "pastor", "imán", "rabbi", "biblia", "corán",
            "oración", "ceremonia", "ritual", "fe", "creencia",
            
            # Filosofía
            "filosofía", "ética", "moral", "lógica", "razón", "pensamiento",
            "existencia", "realidad", "verdad", "conocimiento", "sabiduría",
            "virtud", "vicio", "libertad", "justicia", "bien", "mal"
        ]
        
        self._load_model()
    
    def _load_model(self):
        """Carga el modelo de clasificación"""
        try:
            # Intentar cargar un modelo pre-entrenado para clasificación binaria
            model_name = "distilbert-base-multilingual-cased"
            
            logger.info(f"Cargando clasificador médico desde {model_name}")
            
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
            self.model = DistilBertForSequenceClassification.from_pretrained(
                model_name,
                num_labels=2  # 0: no médico, 1: médico
            )
            
            # Mover modelo al dispositivo
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Clasificador médico cargado exitosamente en {self.device}")
            
        except Exception as e:
            logger.error(f"Error cargando modelo de clasificación: {str(e)}")
            logger.info("Usando clasificación basada en palabras clave como fallback")
            self.model = None
    
    def _keyword_classification(self, text: str) -> Tuple[bool, float]:
        """Clasificación basada en palabras clave como fallback"""
        text_lower = text.lower()
        
        medical_count = sum(1 for keyword in self.medical_keywords if keyword in text_lower)
        non_medical_count = sum(1 for keyword in self.non_medical_keywords if keyword in text_lower)
        
        total_medical = len(self.medical_keywords)
        total_non_medical = len(self.non_medical_keywords)
        
        # Normalizar por el número total de palabras clave
        medical_score = medical_count / total_medical if total_medical > 0 else 0
        non_medical_score = non_medical_count / total_non_medical if total_non_medical > 0 else 0
        
        # Si no hay palabras clave de ningún tipo, considerar como no médico
        if medical_count == 0 and non_medical_count == 0:
            return False, 0.5
        
        # Calcular confianza
        total_score = medical_score + non_medical_score
        if total_score == 0:
            return False, 0.5
        
        confidence = max(medical_score, non_medical_score) / total_score
        is_medical = medical_score > non_medical_score
        
        return is_medical, confidence
    
    def classify(self, text: str) -> Tuple[bool, float]:
        """
        Clasifica si el texto es médico o no
        
        Args:
            text: Texto a clasificar
            
        Returns:
            Tuple[bool, float]: (es_médico, confianza)
        """
        if not text or len(text.strip()) == 0:
            return False, 0.0
        
        # Si el modelo no está disponible, usar clasificación por palabras clave
        if self.model is None:
            return self._keyword_classification(text)
        
        try:
            # Tokenizar el texto
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Realizar predicción
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                
                # Obtener probabilidades
                non_medical_prob = probabilities[0][0].item()
                medical_prob = probabilities[0][1].item()
                
                # Determinar si es médico (probabilidad > 0.5)
                is_medical = medical_prob > 0.5
                confidence = max(medical_prob, non_medical_prob)
                
                logger.debug(f"Clasificación: médico={is_medical}, confianza={confidence:.3f}, "
                           f"prob_médico={medical_prob:.3f}, prob_no_médico={non_medical_prob:.3f}")
                
                return is_medical, confidence
                
        except Exception as e:
            logger.error(f"Error en clasificación con modelo: {str(e)}")
            logger.info("Usando clasificación por palabras clave como fallback")
            return self._keyword_classification(text)
    
    def is_medical_query(self, text: str) -> bool:
        """
        Versión simplificada que solo retorna si es médico o no
        
        Args:
            text: Texto a clasificar
            
        Returns:
            bool: True si es médico, False si no
        """
        is_medical, confidence = self.classify(text)
        
        # Si la confianza es baja, ser más conservador y considerar como no médico
        if confidence < 0.6:
            logger.debug(f"Confianza baja ({confidence:.3f}), considerando como no médico")
            return False
        
        return is_medical

# Instancia global del clasificador
medical_classifier = None

def get_medical_classifier() -> MedicalClassifier:
    """Obtiene la instancia global del clasificador médico"""
    global medical_classifier
    if medical_classifier is None:
        medical_classifier = MedicalClassifier()
    return medical_classifier

def is_medical_query(text: str) -> bool:
    """
    Función de conveniencia para clasificar consultas médicas
    
    Args:
        text: Texto a clasificar
        
    Returns:
        bool: True si es médico, False si no
    """
    classifier = get_medical_classifier()
    return classifier.is_medical_query(text) 