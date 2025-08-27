import os
import json
import logging
from typing import Generator
from transformers import TextIteratorStreamer
from threading import Thread
import torch

# Configuración de logging
logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "Eres LucasMed, un asistente médico de IA especializado en apoyar a médicos en la práctica clínica.\n"
    "\n"
    "⚠️ REGLA PRINCIPAL, ABSOLUTA Y PRIORITARIA ⚠️\n"
    "SOLO puedes responder a preguntas relacionadas con medicina y salud. "
    "Si el tema NO es médico, debes responder ÚNICAMENTE con la frase fija: "
    "'Soy un asistente médico especializado. Solo puedo responder a consultas médicas.'\n"
    "No intentes dar contexto adicional, explicaciones ni disculpas. "
    "No reformules esta frase. No inventes otra variante. "
    "Esta regla es PRIORITARIA sobre cualquier otra.\n"
    "\n"
    "SOLO PUEDES responder a preguntas relacionadas con:\n"
    "✅ Temas permitidos:\n"
    "- Diagnósticos médicos\n"
    "- Tratamientos médicos\n"
    "- Fisiología y anatomía\n"
    "- Farmacología\n"
    "- Análisis clínicos\n"
    "- Síntomas y enfermedades\n"
    "- Procedimientos médicos\n"
    "\n"
    "❌ Temas prohibidos:\n"
    "- Política, economía, deportes\n"
    "- Historia, geografía, entretenimiento\n"
    "- Cualquier tema NO médico\n"
    "\n"
    "- Responde SIEMPRE en español.\n"
    "- Utiliza lenguaje profesional y técnico, adecuado para médicos.\n"
    "- Sé claro, preciso y completo, aportando información relevante para la toma de decisiones clínicas.\n"
    "- Estructura tus respuestas médicas de manera lógica: "
    "1) Resumen breve del punto clave, "
    "2) Explicación clínica detallada, "
    "3) Diagnósticos diferenciales o consideraciones adicionales, "
    "4) Recomendaciones basadas en evidencia o guías clínicas cuando sea posible.\n"
    "- Incluye información adicional relevante: fisiopatología, protocolos de manejo, tratamientos de primera línea, "
    "indicaciones y contraindicaciones, prevención y pronóstico.\n"
    "- Responde solo al último mensaje del usuario, usando contexto si es necesario, pero no repitas respuestas previas.\n"
    "- Aclara cuando la evidencia no sea concluyente o se requiera criterio clínico individualizado.\n"
    "- Señala cuando sea necesario confirmar en guías locales, consensos clínicos o protocolos hospitalarios.\n"
)



def get_system_prompt() -> str:
    """Obtiene el prompt del sistema desde variables de entorno o usa el default"""
    return os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)

def get_medical_image_prompt() -> str:
    return (
        "Eres LucasMed, un experto radiólogo y analista de imágenes médicas, especializado en apoyar a médicos en la "
        "interpretación de estudios diagnósticos.\n"
        "\n"
            "⚠️ REGLA PRINCIPAL, ABSOLUTA Y PRIORITARIA ⚠️\n"
        "SOLO puedes responder a preguntas relacionadas con medicina y salud. "
        "Si el tema NO es médico, debes responder ÚNICAMENTE con la frase fija: "
        "'Soy un asistente médico especializado. Solo puedo responder a consultas médicas.'\n"
        "No intentes dar contexto adicional, explicaciones ni disculpas. "
        "No reformules esta frase. No inventes otra variante. "
        "Esta regla es PRIORITARIA sobre cualquier otra.\n"
        "SOLO PUEDES responder a preguntas relacionadas con:\n"
        "- Medicina y salud\n"
        "- Diagnósticos médicos\n"
        "- Tratamientos médicos\n"
        "- Fisiología y anatomía\n"
        "- Farmacología\n"
        "- Análisis clínicos\n"
        "- Síntomas y enfermedades\n"
        "- Procedimientos médicos\n"
        "- Interpretación de imágenes médicas\n"
        "\n"
        "❌ NO PUEDES responder a:\n"
        "- Política, economía, deportes\n"
        "- Historia, geografía, entretenimiento\n"
        "- Cualquier tema NO médico\n"
        "\n"
        "Si te preguntan algo NO médico, responde ÚNICAMENTE:\n"
        "'Soy un asistente médico especializado. Solo puedo ayudarte con consultas relacionadas con medicina y salud. Por favor, consulta con un especialista en el tema que necesitas.'\n"
        "\n"
        "- Responde SIEMPRE en español.\n"
        "- Utiliza un lenguaje técnico y preciso, adecuado para profesionales de la salud.\n"
        "- Estructura tus análisis en el siguiente orden: "
        "1) Descripción general de la imagen, "
        "2) Hallazgos específicos relevantes, "
        "3) Diagnósticos diferenciales posibles, "
        "4) Recomendaciones (estudios adicionales, correlación clínica o manejo sugerido según guías).\n"
        "- Destaca hallazgos radiológicos de importancia clínica y su posible correlación con la historia del paciente.\n"
        "- Si la imagen es ambigua, indica las limitaciones diagnósticas y qué estudios adicionales podrían aclarar los hallazgos.\n"
        "- Si la imagen no es médica o no puede interpretarse adecuadamente, indícalo claramente.\n"
        "- Siempre que sea posible, menciona guías clínicas relevantes o criterios radiológicos reconocidos "
        "(ej. BI-RADS, Fleischner Society, ACR, etc.).\n"
    )

def get_exam_report_prompt() -> str:
    return (
        "Eres LucasMed, un radiólogo experto especializado en análisis de imágenes médicas.\n"
        "\n"
        "⚠️ REGLA PRINCIPAL ⚠️\n"
        "Si la imagen NO es médica o no puede interpretarse, responde ÚNICAMENTE con esta frase exacta:\n"
        "\"Esta imagen no corresponde a un estudio médico válido o no puede ser interpretada adecuadamente.\"\n"
        "No agregues nada más en ese caso.\n"
        "\n"
        "⚠️ FORMATO DE RESPUESTA OBLIGATORIO (solo para imágenes médicas válidas) ⚠️\n"
        "Responde ÚNICAMENTE con un JSON válido y parseable. NO incluyas texto adicional, explicaciones, comentarios ni bloques de código.\n"
        "El JSON debe contener exclusivamente los siguientes 2 campos:\n"
        "{\n"
        '  "summary": "Análisis técnico detallado y comprehensivo del examen en español, incluyendo descripción de la técnica, calidad de la imagen, evaluación sistemática de todas las estructuras anatómicas relevantes, hallazgos normales y anormales, limitaciones técnicas, correlación clínica cuando sea apropiada, y recomendaciones para estudios adicionales si son necesarios. El resumen debe ser extenso (mínimo 300-500 palabras) y cubrir todos los aspectos relevantes del estudio.",\n'
        '  "findings": ["Hallazgo 1 en español", "Hallazgo 2 en español", "Hallazgo 3 en español"]\n'
        "}\n"
        "\n"
        "⚠️ REGLAS CRÍTICAS ⚠️\n"
        "- Responde SIEMPRE en español.\n"
        "- No inventes claves adicionales ni cambies los nombres existentes.\n"
        "- El campo 'summary' debe ser un string extenso (mínimo 300-500 palabras) con análisis completo y detallado del examen, incluyendo técnica, calidad, evaluación sistemática, hallazgos, limitaciones y recomendaciones.\n"
        "- El campo 'findings' debe ser un array de strings, cada hallazgo como un string independiente.\n"
        "- No uses valores null, listas anidadas ni otros tipos de datos.\n"
        "- No uses bloques de código markdown (```json o ```).\n"
        "- No agregues explicaciones ni texto fuera del JSON.\n"
        "- Valida internamente que el JSON sea correcto antes de responder.\n"
        "- Si hay limitaciones diagnósticas, inclúyelas dentro del campo 'summary' o en un hallazgo dentro de 'findings'.\n"
        "- NO incluyas el campo 'disclaimer' - se agregará automáticamente.\n"
        "- IMPORTANTE: Si usas comillas dentro de los textos, escápalas con \\ (ej: \"texto con \\\"comillas\\\" internas\").\n"
        "- Asegúrate de que todos los strings estén correctamente cerrados con comillas dobles.\n"
        "\n"
        "⚠️ EJEMPLO DE RESPUESTA CORRECTA ⚠️\n"
        "{\n"
        '  "summary": "Se presenta estudio de Tomografía Computarizada (TC) de cráneo en corte sagital, realizado con técnica helicoidal y reconstrucción multiplanar. La calidad técnica de la imagen es adecuada, con buena resolución espacial y contraste tisular apropiado para la evaluación de estructuras intracraneales. La evaluación general revela una morfología cerebral conservada, con adecuada diferenciación entre la sustancia gris y blanca, sin evidencia de cambios atróficos significativos para la edad del paciente. No se identifican lesiones focales, hemorragias intracraneales, ni signos de edema cerebral o efecto de masa. El sistema ventricular es de tamaño normal, con configuración anatómica conservada, sin indicación de hidrocefalia comunicante o no comunicante. Los ventrículos laterales, tercer ventrículo y acueducto de Silvio se visualizan con calibres normales. La fosa posterior muestra un cerebelo de tamaño y morfología normales, con hemisferios cerebelosos simétricos y vermis bien definido. El tronco encefálico se observa sin alteraciones aparentes, con mesencéfalo, protuberancia y bulbo raquídeo de configuración anatómica normal. Las estructuras óseas del cráneo visualizadas en el corte actual no muestran evidencia de fracturas, lesiones líticas o blásticas, ni cambios en la densidad mineral ósea. Los tejidos blandos extracraneales no presentan anomalías significativas, sin evidencia de masas, colecciones o cambios inflamatorios. La evaluación de los senos paranasales en el corte actual no revela opacidades o niveles hidroaéreos patológicos. Se recomienda correlación clínica para interpretación completa de los hallazgos y consideración de estudios adicionales si existe sospecha clínica de patología específica.",\n'
        '  "findings": [\n'
        '    "Cerebro: Estructura general del cerebro dentro de los límites normales. Presencia de surcos y cisuras corticales de apariencia usual para la edad del paciente",\n'
        '    "Ventrículos: Sistema ventricular de tamaño y morfología conservados, sin evidencia de hidrocefalia",\n'
        '    "Sustancia Blanca: Aparente integridad de la sustancia blanca, sin focos de señal anormal evidentes",\n'
        '    "Fosa Posterior: El cerebelo y el tronco encefálico se visualizan sin alteraciones significativas",\n'
        '    "Hueso: Las estructuras óseas del cráneo se observan intactas en el corte actual",\n'
        '    "Tejidos Blandos: Tejidos blandos extracraneales sin particularidades notables"\n'
        '  ]\n'
        "}\n"
    )

def get_diagnosis_prompt() -> str:
    return (
        "Eres LucasMed, un asistente médico de IA especializado en diagnóstico diferencial que ayuda a DOCTORES a explorar diagnósticos.\n"
        "\n"
        "⚠️ REGLA PRINCIPAL, ABSOLUTA Y PRIORITARIA ⚠️\n"
        "SOLO puedes analizar información médica y generar diagnósticos diferenciales. "
        "Si la información NO es médica o no puede interpretarse, debes responder ÚNICAMENTE con la frase fija: "
        "'La información proporcionada no corresponde a datos médicos válidos o no puede ser interpretada adecuadamente.'\n"
        "No intentes dar contexto adicional, explicaciones ni disculpas. "
        "No reformules esta frase. No inventes otra variante. "
        "Esta regla es PRIORITARIA sobre cualquier otra.\n"
        "\n"
        "⚠️ FORMATO DE RESPUESTA OBLIGATORIO ⚠️\n"
        "Responde ÚNICAMENTE con un JSON válido y parseable. NO incluyas texto adicional, explicaciones, comentarios ni bloques de código.\n"
        "El JSON debe contener exclusivamente los siguientes 2 campos:\n"
        "{\n"
        '  "diagnosticos": [\n'
        '    {\n'
        '      "condicion": "Nombre de la condición médica",\n'
        '      "probabilidad": 85,\n'
        '      "justificacion": "Justificación detallada y comprehensiva basada en los datos clínicos proporcionados, incluyendo correlación con síntomas, signos y hallazgos, considerando la fisiopatología y presentación clínica típica de la condición",\n'
        '      "recomendacion": "Recomendación clínica extensa y específica para el médico, incluyendo estudios diagnósticos sugeridos, criterios de derivación, consideraciones terapéuticas iniciales, seguimiento recomendado y precauciones especiales",\n'
        '      "tipo": "obvio"\n'
        '    }\n'
        '  ],\n'
        '  "disclaimer": "Importante: Esta es una sugerencia generada por IA y no reemplaza el juicio clínico profesional. El diagnóstico definitivo y el tratamiento deben ser realizados por un médico."\n'
        "}\n"
        "\n"
        "⚠️ REGLAS CRÍTICAS ⚠️\n"
        "- Responde SIEMPRE en español.\n"
        "- No inventes claves adicionales ni cambies los nombres existentes.\n"
        "- El campo 'diagnosticos' debe ser un array de 4-5 objetos con las claves: condicion, probabilidad, justificacion, recomendacion, tipo.\n"
        "- La 'probabilidad' debe ser un número entero entre 1 y 100.\n"
        "- El 'tipo' debe ser 'obvio' para diagnósticos comunes o 'raro' para diagnósticos menos frecuentes.\n"
        "- La 'justificacion' debe ser extensa (mínimo 100-150 palabras) explicando detalladamente por qué se considera ese diagnóstico, incluyendo correlación con síntomas, signos y hallazgos, fisiopatología y presentación clínica.\n"
        "- La 'recomendacion' debe ser extensa (mínimo 100-150 palabras) y específica para el médico, incluyendo estudios diagnósticos, criterios de derivación, consideraciones terapéuticas, seguimiento y precauciones.\n"
        "- NO incluyas el campo 'disclaimer' - se agregará automáticamente.\n"
        "- IMPORTANTE: Si usas comillas dentro de los textos, escápalas con \\ (ej: \"texto con \\\"comillas\\\" internas\").\n"
        "- Asegúrate de que todos los strings estén correctamente cerrados con comillas dobles.\n"
        "- No uses valores null, listas anidadas ni otros tipos de datos.\n"
        "- No uses bloques de código markdown (```json o ```).\n"
        "- No agregues explicaciones ni texto fuera del JSON.\n"
        "- Valida internamente que el JSON sea correcto antes de responder.\n"
        "- Si el modo es 'obvios', prioriza diagnósticos comunes y frecuentes.\n"
        "- Si el modo es 'raros', prioriza diagnósticos menos frecuentes o atípicos.\n"
        "- No repitas diagnósticos similares en la misma lista.\n"
        "- No pidas datos personales ni hables al paciente, solo al médico.\n"
        "\n"
        "⚠️ EJEMPLO DE RESPUESTA CORRECTA ⚠️\n"
        "{\n"
        '  "diagnosticos": [\n'
        '    {\n'
        '      "condicion": "Neumonía bacteriana",\n'
        '      "probabilidad": 75,\n'
        '      "justificacion": "La combinación de fiebre alta, tos productiva, leucocitosis y PCR elevada es altamente sugestiva de neumonía bacteriana. La fiebre indica respuesta inflamatoria sistémica, la tos productiva sugiere infección del parénquima pulmonar, y los hallazgos de laboratorio (leucocitosis con desviación izquierda y PCR elevada) confirman la presencia de un proceso infeccioso bacteriano activo. La presentación clínica es consistente con la fisiopatología típica de neumonía adquirida en la comunidad, donde los patógenos más frecuentes incluyen Streptococcus pneumoniae, Haemophilus influenzae y Mycoplasma pneumoniae.",\n'
        '      "recomendacion": "Solicitar radiografía de tórax posteroanterior y lateral para confirmar el diagnóstico y evaluar la extensión del compromiso pulmonar. Realizar hemocultivos antes de iniciar antibióticos para identificar el patógeno causal. Iniciar antibióticos empíricos según guías locales (amoxicilina-clavulánico o macrólidos). Considerar hospitalización si hay criterios de gravedad (edad >65 años, comorbilidades, hipoxemia, inestabilidad hemodinámica). Programar seguimiento en 48-72 horas para evaluar respuesta al tratamiento y ajustar terapia según cultivos y evolución clínica.",\n'
        '      "tipo": "obvio"\n'
        '    },\n'
        '    {\n'
        '      "condicion": "Influenza",\n'
        '      "probabilidad": 65,\n'
        '      "justificacion": "La presentación de fiebre de inicio súbito, tos seca y síntomas sistémicos (dolor de cabeza, mialgias) es característica de influenza, especialmente durante la temporada de gripe. La fiebre alta de aparición brusca es un signo cardinal de influenza, diferenciándola de otras infecciones respiratorias virales. La tos seca y la ausencia de síntomas de vías respiratorias superiores prominentes son típicos de la presentación clínica de influenza. Los síntomas sistémicos como dolor de cabeza y mialgias reflejan la respuesta inflamatoria sistémica característica de esta infección viral.",\n'
        '      "recomendacion": "Realizar test rápido de influenza A/B para confirmar el diagnóstico, especialmente si está en ventana terapéutica (primeras 48 horas). Considerar tratamiento con oseltamivir o zanamivir si se confirma influenza y el paciente presenta factores de riesgo para complicaciones. Implementar medidas de aislamiento respiratorio para prevenir transmisión. Recomendar reposo, hidratación adecuada y antipiréticos. Programar seguimiento en 24-48 horas para evaluar evolución y detectar complicaciones tempranas como neumonía viral o bacteriana secundaria.",\n'
        '      "tipo": "obvio"\n'
        '    },\n'
        '    {\n'
        '      "condicion": "Bronquitis aguda",\n'
        '      "probabilidad": 55,\n'
        '      "justificacion": "La tos como síntoma principal, especialmente si es productiva, junto con fiebre leve y ausencia de hallazgos de consolidación pulmonar, sugiere bronquitis aguda. La tos es el síntoma más prominente y puede persistir por 2-3 semanas. La fiebre suele ser de baja intensidad y autolimitada. La ausencia de signos de consolidación pulmonar (como matidez a la percusión o crepitantes localizados) ayuda a diferenciar de neumonía. La bronquitis aguda es típicamente de etiología viral, aunque puede complicarse con sobreinfección bacteriana.",\n'
        '      "recomendacion": "Confirmar ausencia de consolidación pulmonar mediante auscultación cuidadosa. La radiografía de tórax no está indicada rutinariamente en bronquitis aguda sin factores de riesgo. Tratamiento sintomático con antitúsicos si la tos es seca y molesta, o mucolíticos si es productiva. Evitar uso innecesario de antibióticos ya que la mayoría de casos son virales. Educar al paciente sobre la evolución natural de la enfermedad (2-3 semanas) y signos de alarma que requieren reevaluación. Considerar espirometría si hay sospecha de asma o EPOC subyacente.",\n'
        '      "tipo": "obvio"\n'
        '    },\n'
        '    {\n'
        '      "condicion": "COVID-19",\n'
        '      "probabilidad": 45,\n'
        '      "justificacion": "La presentación de fiebre, tos y síntomas sistémicos como dolor de cabeza es compatible con COVID-19, especialmente en contexto de circulación viral activa. La fiebre es un síntoma frecuente en COVID-19, aunque puede estar ausente en algunos casos. La tos seca es característica de esta infección viral. Los síntomas sistémicos como dolor de cabeza y fatiga son comunes en COVID-19. La presentación clínica puede ser similar a influenza, por lo que es importante considerar el contexto epidemiológico y realizar pruebas específicas para diferenciar.",\n'
        '      "recomendacion": "Realizar test de antígenos o PCR para SARS-CoV-2 para confirmar el diagnóstico. Implementar medidas de aislamiento respiratorio estricto hasta confirmar o descartar COVID-19. Evaluar factores de riesgo para enfermedad grave (edad avanzada, comorbilidades, inmunosupresión). Considerar tratamiento con antivirales específicos (nirmatrelvir/ritonavir, remdesivir) en pacientes de alto riesgo. Monitorear saturación de oxígeno y signos de alarma. Programar seguimiento cercano para detectar deterioro respiratorio temprano.",\n'
        '      "tipo": "obvio"\n'
        '    },\n'
        '    {\n'
        '      "condicion": "Sinusitis aguda",\n'
        '      "probabilidad": 35,\n'
        '      "justificacion": "La combinación de fiebre, dolor de cabeza y síntomas respiratorios puede indicar sinusitis aguda, especialmente si hay síntomas de vías respiratorias superiores asociados. El dolor de cabeza puede ser referido desde los senos paranasales inflamados. La fiebre indica proceso infeccioso activo. La sinusitis aguda puede presentarse con síntomas sistémicos similares a otras infecciones respiratorias. Es importante evaluar la presencia de síntomas específicos de sinusitis como dolor facial, congestión nasal y secreción purulenta.",\n'
        '      "recomendacion": "Realizar examen físico completo de cabeza y cuello, incluyendo palpación de senos paranasales y rinoscopia anterior. Considerar radiografía de senos paranasales solo si el diagnóstico es incierto. Tratamiento inicial con descongestionantes nasales y analgésicos. Considerar antibióticos solo si hay síntomas persistentes por más de 10 días o empeoramiento después de 5-7 días. Educar sobre técnicas de irrigación nasal y medidas de higiene. Programar seguimiento en 7-10 días para evaluar respuesta al tratamiento.",\n'
        '      "tipo": "obvio"\n'
        '    }\n'
        '  ]\n'
        "}\n"
    )

def clean_json_response(response: str) -> str:
    """Limpia una respuesta para extraer JSON válido, manejando diferentes formatos"""
    if not response:
        return ""
    
    response_clean = response.strip()
    
    # Remover bloques de código markdown
    if response_clean.startswith('```json'):
        # Remover ```json al inicio
        response_clean = response_clean[7:]
    elif response_clean.startswith('```'):
        # Remover ``` al inicio
        response_clean = response_clean[3:]
    
    # Remover ``` al final si existe
    if response_clean.endswith('```'):
        response_clean = response_clean[:-3]
    
    # Buscar el último JSON válido (el más reciente, que debería ser la respuesta del modelo)
    # Buscar todos los pares de { y } para encontrar el JSON más completo
    import re
    
    # Encontrar todos los pares de llaves
    brace_pairs = []
    stack = []
    for i, char in enumerate(response_clean):
        if char == '{':
            stack.append(i)
        elif char == '}':
            if stack:
                start = stack.pop()
                brace_pairs.append((start, i + 1))
    
    # Si no hay pares de llaves, buscar el primer JSON simple
    if not brace_pairs:
        start_idx = response_clean.find('{')
        end_idx = response_clean.rfind('}') + 1
        if start_idx != -1 and end_idx > start_idx:
            json_str = response_clean[start_idx:end_idx]
        else:
            return response_clean.strip()
    else:
        # Tomar el último par de llaves (el más reciente)
        start_idx, end_idx = brace_pairs[-1]
        json_str = response_clean[start_idx:end_idx]
        logger.info(f"Seleccionado JSON #{len(brace_pairs)} (último) de {len(brace_pairs)} encontrados")
        
        # Verificar si el JSON está completo
        if not json_str.strip().endswith('}'):
            logger.warning("JSON parece estar incompleto (no termina con '}')")
            # Intentar completar el JSON si es posible
            if '"findings":' in json_str and not json_str.strip().endswith(']'):
                # Si tiene findings pero no está cerrado, intentar completarlo
                json_str = json_str.strip() + ']}'
                logger.info("JSON completado automáticamente")
        
        # Intentar arreglar problemas comunes de JSON
        try:
            # Primero intentar parsear como está
            json.loads(json_str)
            logger.info("JSON válido sin necesidad de arreglos")
            return json_str.strip()
        except json.JSONDecodeError as e:
            logger.warning(f"JSON inválido detectado, intentando arreglar: {str(e)}")
            logger.warning(f"JSON problemático: {json_str}")
            
            # Intentar arreglar comillas no escapadas en strings
            try:
                # Buscar strings que contengan comillas dobles sin escapar
                import re
                
                # Reemplazar comillas dobles internas con comillas simples
                fixed_json = re.sub(r'(?<!\\)"(?=.*":\s*"[^"]*$)', "'", json_str)
                
                # Intentar parsear el JSON arreglado
                json.loads(fixed_json)
                logger.info("JSON arreglado exitosamente")
                return fixed_json.strip()
                
            except (json.JSONDecodeError, Exception) as e2:
                logger.error(f"No se pudo arreglar el JSON: {str(e2)}")
                # Si no se puede arreglar, devolver el original para que el parser maneje el error
                return json_str.strip()
    
    return response_clean.strip()

class ExamReportOutputParser:
    """Parser para extraer y validar reportes de examen médico del modelo"""
    
    def __init__(self):
        self.required_keys = ['summary', 'findings']
        self.disclaimer = "Importante: Este es un análisis preliminar generado por IA y no debe considerarse un diagnóstico médico definitivo. La interpretación de imágenes médicas es compleja y debe ser realizada por un radiólogo certificado. Consulte a un profesional de la salud para una evaluación completa y un diagnóstico preciso."
    
    def parse(self, response: str) -> dict:
        """
        Parsea la respuesta del modelo y extrae el reporte de examen
        
        Args:
            response: Respuesta del modelo
            
        Returns:
            dict: Reporte parseado con las claves requeridas + disclaimer
        """
        try:
            # Limpiar la respuesta
            json_str = clean_json_response(response)
            
            if not json_str:
                logger.warning("No se encontró JSON en la respuesta")
                return self._create_fallback_response("No se encontró JSON válido en la respuesta")
            
            # Log para debugging
            logger.info(f"JSON a parsear (longitud: {len(json_str)}): {json_str}")
            
            # Parsear JSON
            report_data = json.loads(json_str)
            
            # Validar y completar claves requeridas
            validated_data = self._validate_and_complete(report_data)
            
            # Agregar disclaimer automáticamente
            validated_data['disclaimer'] = self.disclaimer
            
            logger.info("JSON parseado exitosamente")
            return validated_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parseando JSON: {str(e)}")
            logger.error(f"JSON problemático: {json_str[:200]}...")
            
            # Intentar extraer información útil del JSON malformado
            try:
                # Buscar patrones básicos en el JSON malformado
                import re
                
                # Buscar el último JSON en la respuesta completa
                all_jsons = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response)
                if all_jsons:
                    # Tomar el último JSON encontrado
                    last_json = all_jsons[-1]
                    logger.info(f"Extrayendo del último JSON encontrado: {last_json[:200]}...")
                    json_str = last_json
                
                # Extraer summary si existe (más robusto)
                summary_match = re.search(r'"summary":\s*"([^"]*)"', json_str)
                if summary_match:
                    summary = summary_match.group(1)
                    # Si el summary está truncado, intentar obtener más
                    if len(summary) < 50:  # Si es muy corto, probablemente está truncado
                        # Buscar el summary completo hasta el siguiente campo
                        full_summary_match = re.search(r'"summary":\s*"([^"]*(?:"[^"]*"[^"]*)*)"', json_str)
                        if full_summary_match:
                            summary = full_summary_match.group(1)
                else:
                    summary = "No se pudo extraer el resumen"
                
                # Extraer findings si existe (más robusto)
                findings_match = re.search(r'"findings":\s*\[(.*?)\]', json_str, re.DOTALL)
                if findings_match:
                    findings_text = findings_match.group(1)
                    # Limpiar y extraer elementos del array
                    findings_items = re.findall(r'"([^"]*)"', findings_text)
                    if findings_items:
                        findings = findings_items
                    else:
                        # Si no se encontraron elementos, intentar extraer texto simple
                        findings = ["Hallazgos extraídos del análisis de la imagen"]
                else:
                    # Si no hay findings, intentar extraer cualquier texto después del summary
                    after_summary = json_str.split('"summary":')[1] if '"summary":' in json_str else ""
                    if after_summary:
                        # Buscar cualquier texto que parezca un hallazgo
                        potential_findings = re.findall(r'"([^"]{10,})"', after_summary)
                        findings = potential_findings[:5] if potential_findings else ["Hallazgos extraídos del análisis de la imagen"]
                    else:
                        findings = ["No se pudieron extraer hallazgos específicos"]
                
                logger.info(f"Se extrajo información parcial: summary ({len(summary)} chars), findings ({len(findings)} items)")
                
                return {
                    'summary': summary,
                    'findings': findings,
                    'disclaimer': self.disclaimer
                }
                
            except Exception as extract_error:
                logger.error(f"Error extrayendo información parcial: {str(extract_error)}")
                return self._create_fallback_response(f"Error en el formato JSON: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error inesperado en el parser: {str(e)}")
            return self._create_fallback_response(f"Error inesperado: {str(e)}")
    
    def _validate_and_complete(self, data: dict) -> dict:
        """Valida y completa las claves requeridas en el reporte"""
        validated = {}
        
        # Validar summary
        if 'summary' in data and data['summary']:
            validated['summary'] = str(data['summary']).strip()
        else:
            validated['summary'] = "No se pudo generar un análisis estructurado de la imagen."
        
        # Validar findings (puede ser array o string)
        if 'findings' in data and data['findings']:
            findings = data['findings']
            if isinstance(findings, list):
                # Si es una lista, validar que todos los elementos sean strings
                validated['findings'] = [str(item).strip() for item in findings if item]
            elif isinstance(findings, str):
                # Si es string, intentar convertirlo a lista o mantener como string
                validated['findings'] = findings.strip()
            else:
                validated['findings'] = "Se requiere revisión manual por un radiólogo certificado."
        else:
            validated['findings'] = "Se requiere revisión manual por un radiólogo certificado."
        
        return validated
    
    def _create_fallback_response(self, error_message: str) -> dict:
        """Crea una respuesta de fallback cuando hay errores"""
        return {
            'summary': f"Error en el procesamiento: {error_message}",
            'findings': "Se requiere revisión manual por un radiólogo certificado.",
            'disclaimer': self.disclaimer
        }
    
    def format_for_api(self, parsed_data: dict) -> str:
        """Formatea los datos parseados para la respuesta de la API"""
        return json.dumps(parsed_data, ensure_ascii=False)

class DiagnosisOutputParser:
    """Parser para extraer y validar diagnósticos diferenciales del modelo"""
    
    def __init__(self):
        self.required_keys = ['diagnosticos']
        self.disclaimer = "Importante: Esta es una sugerencia generada por IA y no reemplaza el juicio clínico profesional. El diagnóstico definitivo y el tratamiento deben ser realizados por un médico."
    
    def parse(self, response: str) -> dict:
        """
        Parsea la respuesta del modelo y extrae los diagnósticos diferenciales
        
        Args:
            response: Respuesta del modelo
            
        Returns:
            dict: Diagnósticos parseados con las claves requeridas + disclaimer
        """
        try:
            # Limpiar la respuesta
            json_str = clean_json_response(response)
            
            if not json_str:
                logger.warning("No se encontró JSON en la respuesta")
                return self._create_fallback_response("No se encontró JSON válido en la respuesta")
            
            # Log para debugging
            logger.info(f"JSON a parsear (longitud: {len(json_str)}): {json_str}")
            
            # Parsear JSON
            diagnosis_data = json.loads(json_str)
            
            # Validar y completar claves requeridas
            validated_data = self._validate_and_complete(diagnosis_data)
            
            # Agregar disclaimer automáticamente
            validated_data['disclaimer'] = self.disclaimer
            
            logger.info("JSON parseado exitosamente")
            return validated_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parseando JSON: {str(e)}")
            logger.error(f"JSON problemático: {json_str[:200]}...")
            
            # Intentar extraer información útil del JSON malformado
            try:
                # Buscar patrones básicos en el JSON malformado
                import re
                
                # Buscar el último JSON en la respuesta completa
                all_jsons = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response)
                if all_jsons:
                    # Tomar el último JSON encontrado
                    last_json = all_jsons[-1]
                    logger.info(f"Extrayendo del último JSON encontrado: {last_json[:200]}...")
                    json_str = last_json
                
                # Extraer diagnósticos si existe
                diagnosticos_match = re.search(r'"diagnosticos":\s*\[(.*?)\]', json_str, re.DOTALL)
                if diagnosticos_match:
                    diagnosticos_text = diagnosticos_match.group(1)
                    # Extraer objetos de diagnóstico individuales
                    diagnosticos = self._extract_diagnosticos(diagnosticos_text)
                else:
                    diagnosticos = [self._create_default_diagnostico()]
                
                logger.info(f"Se extrajo información parcial: {len(diagnosticos)} diagnósticos")
                
                return {
                    'diagnosticos': diagnosticos,
                    'disclaimer': self.disclaimer
                }
                
            except Exception as extract_error:
                logger.error(f"Error extrayendo información parcial: {str(extract_error)}")
                return self._create_fallback_response(f"Error en el formato JSON: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error inesperado en el parser: {str(e)}")
            return self._create_fallback_response(f"Error inesperado: {str(e)}")
    
    def _validate_and_complete(self, data: dict) -> dict:
        """Valida y completa las claves requeridas en el diagnóstico"""
        validated = {}
        
        # Validar diagnósticos
        if 'diagnosticos' in data and isinstance(data['diagnosticos'], list):
            validated_diagnosticos = []
            for diagnostico in data['diagnosticos']:
                if isinstance(diagnostico, dict):
                    validated_diagnostico = self._validate_diagnostico(diagnostico)
                    validated_diagnosticos.append(validated_diagnostico)
            
            if validated_diagnosticos:
                validated['diagnosticos'] = validated_diagnosticos
            else:
                validated['diagnosticos'] = [self._create_default_diagnostico()]
        else:
            validated['diagnosticos'] = [self._create_default_diagnostico()]
        
        return validated
    
    def _validate_diagnostico(self, diagnostico: dict) -> dict:
        """Valida un diagnóstico individual"""
        validated = {}
        
        # Validar condición
        if 'condicion' in diagnostico and diagnostico['condicion']:
            validated['condicion'] = str(diagnostico['condicion']).strip()
        else:
            validated['condicion'] = "Diagnóstico no especificado"
        
        # Validar probabilidad
        if 'probabilidad' in diagnostico:
            try:
                prob = int(diagnostico['probabilidad'])
                validated['probabilidad'] = max(1, min(100, prob))  # Asegurar rango 1-100
            except (ValueError, TypeError):
                validated['probabilidad'] = 50
        else:
            validated['probabilidad'] = 50
        
        # Validar justificación
        if 'justificacion' in diagnostico and diagnostico['justificacion']:
            validated['justificacion'] = str(diagnostico['justificacion']).strip()
        else:
            validated['justificacion'] = "Justificación no disponible"
        
        # Validar recomendación
        if 'recomendacion' in diagnostico and diagnostico['recomendacion']:
            validated['recomendacion'] = str(diagnostico['recomendacion']).strip()
        else:
            validated['recomendacion'] = "Recomendación no disponible"
        
        # Validar tipo
        if 'tipo' in diagnostico and diagnostico['tipo'] in ['obvio', 'raro']:
            validated['tipo'] = diagnostico['tipo']
        else:
            validated['tipo'] = 'obvio'
        
        return validated
    
    def _extract_diagnosticos(self, diagnosticos_text: str) -> list:
        """Extrae diagnósticos de texto JSON malformado"""
        import re
        
        # Buscar objetos de diagnóstico individuales
        diagnostico_pattern = r'\{[^{}]*"condicion"[^{}]*\}'
        matches = re.findall(diagnostico_pattern, diagnosticos_text)
        
        diagnosticos = []
        for match in matches:
            try:
                # Intentar parsear cada diagnóstico individual
                diagnostico_data = json.loads(match)
                validated_diagnostico = self._validate_diagnostico(diagnostico_data)
                diagnosticos.append(validated_diagnostico)
            except:
                continue
        
        if not diagnosticos:
            diagnosticos = [self._create_default_diagnostico()]
        
        return diagnosticos
    
    def _create_default_diagnostico(self) -> dict:
        """Crea un diagnóstico por defecto"""
        return {
            'condicion': 'Diagnóstico diferencial no disponible',
            'probabilidad': 50,
            'justificacion': 'No se pudo generar un análisis diferencial con la información proporcionada',
            'recomendacion': 'Se requiere evaluación clínica completa por un médico',
            'tipo': 'obvio'
        }
    
    def _create_fallback_response(self, error_message: str) -> dict:
        """Crea una respuesta de fallback cuando hay errores"""
        return {
            'diagnosticos': [{
                'condicion': f'Error en el procesamiento: {error_message}',
                'probabilidad': 0,
                'justificacion': 'No se pudo procesar la información médica proporcionada',
                'recomendacion': 'Se requiere evaluación clínica directa por un médico',
                'tipo': 'obvio'
            }],
            'disclaimer': self.disclaimer
        }
    
    def format_for_api(self, parsed_data: dict) -> str:
        """Formatea los datos parseados para la respuesta de la API"""
        return json.dumps(parsed_data, ensure_ascii=False)

def clean_response(full_response, prompt):
    """Limpia la respuesta removiendo el prompt original y repeticiones"""
    # Buscar el último token de asistente en el prompt
    assistant_markers = ["<|im_start|>assistant", "<|im_end|>", "<|im_start|>user", "<|im_end|>"]
    
    # Intentar diferentes estrategias de limpieza
    cleaned = full_response
    
    # Estrategia 1: Buscar después del último marcador de asistente
    for marker in assistant_markers:
        if marker in prompt:
            parts = prompt.split(marker)
            if len(parts) > 1:
                prompt_until_assistant = parts[0] + marker
                if prompt_until_assistant in full_response:
                    cleaned = full_response.split(prompt_until_assistant)[-1]
                    break
    
    # Estrategia 2: Si no funciona, buscar después de "assistant"
    if cleaned == full_response and "assistant" in prompt.lower():
        assistant_index = prompt.lower().find("assistant")
        if assistant_index != -1:
            prompt_until_assistant = prompt[:assistant_index] + "assistant"
            if prompt_until_assistant in full_response:
                cleaned = full_response.split(prompt_until_assistant)[-1]
    
    # Estrategia 3: Buscar después de "model" (común en algunos modelos)
    if cleaned == full_response and "model" in full_response.lower():
        model_index = full_response.lower().find("model")
        if model_index != -1:
            cleaned = full_response[model_index + 5:]  # "model" tiene 5 caracteres
    
    # Estrategia 4: Buscar después de "Responde al último mensaje"
    if cleaned == full_response and "Responde al último mensaje" in prompt:
        respond_index = full_response.find("Responde al último mensaje")
        if respond_index != -1:
            # Buscar después de esta frase en la respuesta
            prompt_after_respond = prompt[respond_index:]
            if prompt_after_respond in full_response:
                cleaned = full_response.split(prompt_after_respond)[-1]
    
    # Estrategia 5: Remover el prompt completo si está al inicio
    if cleaned == full_response and prompt in full_response:
        cleaned = full_response.replace(prompt, "")
    
    # Estrategia 6: Buscar el inicio real de la respuesta (después del prompt)
    if cleaned == full_response:
        # Buscar patrones comunes que indican el inicio de la respuesta
        response_patterns = [
            "Hola, lamento escuchar",
            "Entiendo que",
            "Para ayudarte mejor",
            "¿Podrías decirme",
            "Te recomiendo",
            "Mientras tanto",
            "¡Hola!",
            "Hola,"
        ]
        
        for pattern in response_patterns:
            if pattern in full_response:
                pattern_index = full_response.find(pattern)
                if pattern_index > len(prompt) * 0.8:  # Si está después del 80% del prompt
                    cleaned = full_response[pattern_index:]
                    break
    
    # Limpiar marcadores de chat si quedan
    for marker in assistant_markers:
        cleaned = cleaned.replace(marker, "")
    
    # Limpiar líneas vacías al inicio
    cleaned = cleaned.lstrip()
    
    # Remover repeticiones excesivas
    sentences = cleaned.split('.')
    if len(sentences) > 2:
        # Mantener solo oraciones únicas
        unique_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence not in unique_sentences:
                unique_sentences.append(sentence)
        
        # Si hay muchas repeticiones, mantener solo las primeras 2-3 oraciones únicas
        if len(unique_sentences) > 3:
            unique_sentences = unique_sentences[:3]
        
        cleaned = '. '.join(unique_sentences) + ('.' if cleaned.endswith('.') else '')
    
    return cleaned.strip()

def is_trivial_question(text: str) -> bool:
    text = text.strip()
    # Preguntas triviales: cortas, directas, sin contexto ni razonamiento
    if len(text) < 30 and text.endswith("?"):
        return True
    # Preguntas tipo saludo o confirmación
    if text.lower() in {"hola", "gracias", "ok", "buenos días"}:
        return True
    return False

def generate_stream_response(model, processor, formatted_prompt, user_input=None, max_new_tokens=1100):
    """Genera respuesta en streaming real usando TextIteratorStreamer"""
    logger.info(f"Iniciando generación con max_new_tokens={max_new_tokens}")
    
    # Procesar con el modelo
    inputs = processor(
        text=formatted_prompt,
        return_tensors="pt"
    ).to("cuda")
    
    logger.info(f"Prompt procesado, tokens de entrada: {len(inputs.input_ids[0])}")

    # Streamer que produce texto incrementalmente
    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    # Parámetros optimizados para respuestas más extensas y detalladas
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,           # Habilitar muestreo para mayor diversidad
        temperature=0.7,          # Temperatura moderada para balance entre creatividad y coherencia
        top_p=0.9,               # Nucleus sampling para mejor calidad
        repetition_penalty=1.1,   # Penalizar repeticiones
        length_penalty=1.0,       # No penalizar respuestas largas
        early_stopping=False,     # Permitir que la respuesta se complete naturalmente
        streamer=streamer,
    )

    # Ejecutar la generación en un hilo separado
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    assistant_markers = ["<|im_start|>assistant", "<|im_end|>", "<|im_start|>user", "<|im_end|>"]
    full_response = ""

    try:
        for new_text in streamer:
            # Limpiar marcadores de chat
            for marker in assistant_markers:
                new_text = new_text.replace(marker, "")
            
            if new_text:
                full_response += new_text
                
                # Detectar repeticiones (menos agresivo)
                sentences = full_response.split('.')
                if len(sentences) > 5:
                    # Verificar si las últimas 5 oraciones son exactamente iguales
                    recent_sentences = sentences[-5:]
                    if len(set(recent_sentences)) == 1 and len(recent_sentences[0].strip()) > 20:
                        # Detener solo si hay repetición excesiva y clara
                        logger.info("Detectada repetición excesiva, deteniendo generación")
                        break
                
                yield f"data: {json.dumps({'token': new_text, 'finished': False})}\n\n"
    finally:
        thread.join()
        logger.info(f"Generación completada. Respuesta total: {len(full_response)} caracteres")

    # Señalizar fin
    yield f"data: {json.dumps({'token': '', 'finished': True})}\n\n"

def generate_stream_response_with_images(model, processor, formatted_prompt, images, user_input=None, max_new_tokens=1100):
    """Genera respuesta en streaming real usando TextIteratorStreamer para contenido multimodal"""
    logger.info(f"Iniciando generación multimodal con max_new_tokens={max_new_tokens}")
    
    # Validar que hay imágenes para procesar
    if not images or len(images) == 0:
        logger.warning("No hay imágenes para procesar, usando generación de texto normal")
        # Si no hay imágenes, usar la función de texto normal
        return generate_stream_response(model, processor, formatted_prompt, user_input, max_new_tokens)
    
    # Procesar con el modelo incluyendo imágenes
    logger.info(f"Procesando {len(images)} imágenes con el modelo")
    inputs = processor(
        text=formatted_prompt,
        images=images,
        return_tensors="pt"
    ).to("cuda")
    
    logger.info(f"Prompt multimodal procesado, tokens de entrada: {len(inputs.input_ids[0])}")
    logger.info(f"Tipo de inputs: {type(inputs)}")
    logger.info(f"Claves de inputs: {inputs.keys() if hasattr(inputs, 'keys') else 'No es un dict'}")

    # Streamer que produce texto incrementalmente
    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    # Parámetros optimizados para respuestas más extensas y detalladas
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,           # Habilitar muestreo para mayor diversidad
        temperature=0.7,          # Temperatura moderada para balance entre creatividad y coherencia
        top_p=0.9,               # Nucleus sampling para mejor calidad
        repetition_penalty=1.1,   # Penalizar repeticiones
        length_penalty=1.0,       # No penalizar respuestas largas
        early_stopping=False,     # Permitir que la respuesta se complete naturalmente
        streamer=streamer,
    )

    # Ejecutar la generación en un hilo separado
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    assistant_markers = ["<|im_start|>assistant", "<|im_end|>", "<|im_start|>user", "<|im_end|>"]
    full_response = ""

    try:
        token_count = 0
        for new_text in streamer:
            # Limpiar marcadores de chat
            for marker in assistant_markers:
                new_text = new_text.replace(marker, "")
            
            if new_text:
                full_response += new_text
                token_count += 1
                
                # Detectar repeticiones (menos agresivo)
                sentences = full_response.split('.')
                if len(sentences) > 5:
                    # Verificar si las últimas 5 oraciones son exactamente iguales
                    recent_sentences = sentences[-5:]
                    if len(set(recent_sentences)) == 1 and len(recent_sentences[0].strip()) > 20:
                        # Detener solo si hay repetición excesiva y clara
                        logger.info("Detectada repetición excesiva, deteniendo generación")
                        break
                
                yield f"data: {json.dumps({'token': new_text, 'finished': False})}\n\n"
        
        # Si no se generó ningún token, enviar un mensaje de error
        if token_count == 0:
            logger.warning("No se generaron tokens, enviando mensaje de error")
            yield f"data: {json.dumps({'token': 'No se pudo generar una respuesta. Por favor, intenta de nuevo.', 'finished': False})}\n\n"
            
    except Exception as e:
        logger.error(f"Error durante la generación multimodal: {str(e)}")
        yield f"data: {json.dumps({'token': f'Error en la generación: {str(e)}', 'finished': False})}\n\n"
    finally:
        thread.join()
        logger.info(f"Generación multimodal completada. Tokens generados: {token_count}, Respuesta total: {len(full_response)} caracteres")

    # Señalizar fin
    yield f"data: {json.dumps({'token': '', 'finished': True})}\n\n"

def clean_context_from_streaming_errors(context: str) -> str:
    """Limpia el contexto de errores de streaming y datos JSON"""
    if not context:
        return context
    
    # Remover líneas que contengan errores de streaming
    lines = context.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Ignorar líneas que contengan errores de streaming
        if any(error_pattern in line.lower() for error_pattern in [
            '{"error":', 'http 500', 'internal server error', 
            'finished:', 'token:', '{"token"'
        ]):
            continue
            
        # Ignorar líneas que empiecen con 'data:' pero NO sean imágenes
        if line.startswith('data:') and not line.startswith('data:image/'):
            continue
            
        # Ignorar líneas que sean solo JSON
        if line.startswith('{') and line.endswith('}'):
            continue
            
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def process_context_messages(context: str) -> list:
    """Procesa el contexto y construye la lista de mensajes dinámicamente"""
    messages = []
    
    # Limpiar el contexto de posibles errores de streaming
    context = clean_context_from_streaming_errors(context)
    
    # Procesar el contexto línea por línea
    context_lines = context.strip().split('\n')
    current_message = None
    current_role = None
    
    for line in context_lines:
        line = line.strip()
        if not line:
            continue
            
        # Detectar si es una imagen (data URI)
        is_image = line.startswith('data:image/')
        
        # Detectar si es inicio de un nuevo mensaje
        # Formato 1: [Usuario] o [Asistente]
        if line.startswith('[Usuario]') or line.startswith('[User]'):
            # Guardar mensaje anterior si existe
            if current_message:
                messages.append(current_message)
            
            # Iniciar nuevo mensaje de usuario
            user_message = line.split(']', 1)[1].strip() if ']' in line else line
            current_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message}
                ]
            }
            current_role = "user"
            
        elif line.startswith('[Asistente]') or line.startswith('[Assistant]'):
            # Guardar mensaje anterior si existe
            if current_message:
                messages.append(current_message)
            
            # Iniciar nuevo mensaje de asistente
            assistant_message = line.split(']', 1)[1].strip() if ']' in line else line
            current_message = {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": assistant_message}
                ]
            }
            current_role = "assistant"
            
        # Formato 2: user: o assistant: (formato original)
        elif line.startswith('user:') or line.startswith('User:') or line.startswith('Usuario:'):
            # Guardar mensaje anterior si existe
            if current_message:
                messages.append(current_message)
            
            # Iniciar nuevo mensaje de usuario
            user_message = line.split(':', 1)[1].strip() if ':' in line else line
            current_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message}
                ]
            }
            current_role = "user"
            
        elif line.startswith('assistant:') or line.startswith('Assistant:') or line.startswith('Asistente:'):
            # Guardar mensaje anterior si existe
            if current_message:
                messages.append(current_message)
            
            # Iniciar nuevo mensaje de asistente
            assistant_message = line.split(':', 1)[1].strip() if ':' in line else line
            current_message = {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": assistant_message}
                ]
            }
            current_role = "assistant"
            
        elif is_image:
            # Es una imagen, convertir data URI a PIL Image
            try:
                # Extraer la parte base64
                header, encoded = line.split(",", 1)
                image_bytes = base64.b64decode(encoded)
                image = Image.open(BytesIO(image_bytes))
                
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Agregar la imagen al mensaje actual o crear uno nuevo
                if current_message and current_role == "user":
                    # Agregar la imagen al mensaje de usuario actual
                    current_message["content"].append({"type": "image", "image": image})
                else:
                    # Crear nuevo mensaje de usuario con la imagen
                    if current_message:
                        messages.append(current_message)
                    
                    current_message = {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image}
                        ]
                    }
                    current_role = "user"
                    
            except Exception as e:
                logger.error(f"Error procesando imagen en contexto: {str(e)}")
                # Si hay error, tratar como texto
                if current_message:
                    text_content = None
                    for content in current_message["content"]:
                        if content["type"] == "text":
                            text_content = content
                            break
                    
                    if text_content:
                        text_content["text"] = f"{text_content['text']}\n{line}"
                    else:
                        current_message["content"].append({"type": "text", "text": line})
                else:
                    current_message = {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": line}
                        ]
                    }
                    current_role = "user"
            
        else:
            # Si no tiene prefijo, es continuación del mensaje actual
            if current_message:
                # Buscar si ya hay contenido de texto
                text_content = None
                for content in current_message["content"]:
                    if content["type"] == "text":
                        text_content = content
                        break
                
                if text_content:
                    # Agregar la línea al texto existente
                    text_content["text"] = f"{text_content['text']}\n{line}"
                else:
                    # Agregar nuevo contenido de texto
                    current_message["content"].append({"type": "text", "text": line})
            else:
                # Si no hay mensaje actual, asumir que es mensaje de usuario
                current_message = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": line}
                    ]
                }
                current_role = "user"
    
    # Agregar el último mensaje si existe
    if current_message:
        messages.append(current_message)
    
    # Asegurar que los roles alternen correctamente
    cleaned_messages = []
    last_role = None
    
    for message in messages:
        current_role = message["role"]
        
        # Si es el primer mensaje, agregarlo
        if last_role is None:
            cleaned_messages.append(message)
            last_role = current_role
            continue
        
        # Si el rol actual es diferente al anterior, agregarlo
        if current_role != last_role:
            cleaned_messages.append(message)
            last_role = current_role
        else:
            # Si hay roles consecutivos iguales, combinar el contenido
            if cleaned_messages:
                # Combinar el contenido del mensaje actual con el último mensaje del mismo rol
                last_message = cleaned_messages[-1]
                
                # Agregar todo el contenido del mensaje actual al último mensaje
                for content in message["content"]:
                    if content["type"] == "text":
                        # Buscar si ya existe contenido de texto
                        existing_text = None
                        for existing_content in last_message["content"]:
                            if existing_content["type"] == "text":
                                existing_text = existing_content
                                break
                        
                        if existing_text:
                            existing_text["text"] = f"{existing_text['text']}\n{content['text']}"
                        else:
                            last_message["content"].append(content)
                    else:
                        # Para imágenes, agregar directamente
                        last_message["content"].append(content)
    
    print("*****************************************cleaned_messages****************************************")
    print(cleaned_messages)
    
    # Contar imágenes en los mensajes
    image_count = 0
    for message in cleaned_messages:
        if message["role"] == "user":
            for content in message["content"]:
                if content["type"] == "image":
                    image_count += 1
    
    logger.info(f"Procesamiento completado. {len(cleaned_messages)} mensajes, {image_count} imágenes encontradas")
    
    return cleaned_messages

def process_context_messages_with_images(context: str) -> list:
    """Procesa el contexto y construye la lista de mensajes dinámicamente, incluyendo imágenes"""
    from PIL import Image
    import base64
    from io import BytesIO
    
    messages = []
    
    # Limpiar el contexto de posibles errores de streaming
    context = clean_context_from_streaming_errors(context)
    
    # Procesar el contexto línea por línea
    context_lines = context.strip().split('\n')
    current_message = None
    current_role = None
    
    for line in context_lines:
        line = line.strip()
        if not line:
            continue
            
        # Detectar si es una imagen (data URI)
        is_image = line.startswith('data:image/')
        
        # Detectar si es inicio de un nuevo mensaje
        # Formato 1: [Usuario] o [Asistente]
        if line.startswith('[Usuario]') or line.startswith('[User]'):
            # Guardar mensaje anterior si existe
            if current_message:
                messages.append(current_message)
            
            # Iniciar nuevo mensaje de usuario
            user_message = line.split(']', 1)[1].strip() if ']' in line else line
            current_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message}
                ]
            }
            current_role = "user"
            
        elif line.startswith('[Asistente]') or line.startswith('[Assistant]'):
            # Guardar mensaje anterior si existe
            if current_message:
                messages.append(current_message)
            
            # Iniciar nuevo mensaje de asistente
            assistant_message = line.split(']', 1)[1].strip() if ']' in line else line
            current_message = {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": assistant_message}
                ]
            }
            current_role = "assistant"
            
        # Formato 2: user: o assistant: (formato original)
        elif line.startswith('user:') or line.startswith('User:') or line.startswith('Usuario:'):
            # Guardar mensaje anterior si existe
            if current_message:
                messages.append(current_message)
            
            # Iniciar nuevo mensaje de usuario
            user_message = line.split(':', 1)[1].strip() if ':' in line else line
            current_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message}
                ]
            }
            current_role = "user"
            
        elif line.startswith('assistant:') or line.startswith('Assistant:') or line.startswith('Asistente:'):
            # Guardar mensaje anterior si existe
            if current_message:
                messages.append(current_message)
            
            # Iniciar nuevo mensaje de asistente
            assistant_message = line.split(':', 1)[1].strip() if ':' in line else line
            current_message = {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": assistant_message}
                ]
            }
            current_role = "assistant"
            
        elif is_image:
            # Es una imagen, convertir data URI a PIL Image
            try:
                # Extraer la parte base64
                header, encoded = line.split(",", 1)
                image_bytes = base64.b64decode(encoded)
                image = Image.open(BytesIO(image_bytes))
                
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Agregar la imagen al mensaje actual o crear uno nuevo
                if current_message and current_role == "user":
                    # Agregar la imagen al mensaje de usuario actual
                    current_message["content"].append({"type": "image", "image": image})
                else:
                    # Crear nuevo mensaje de usuario con la imagen
                    if current_message:
                        messages.append(current_message)
                    
                    current_message = {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image}
                        ]
                    }
                    current_role = "user"
                    
            except Exception as e:
                logger.error(f"Error procesando imagen en contexto: {str(e)}")
                # Si hay error, tratar como texto
                if current_message:
                    text_content = None
                    for content in current_message["content"]:
                        if content["type"] == "text":
                            text_content = content
                            break
                    
                    if text_content:
                        text_content["text"] = f"{text_content['text']}\n{line}"
                    else:
                        current_message["content"].append({"type": "text", "text": line})
                else:
                    current_message = {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": line}
                        ]
                    }
                    current_role = "user"
            
        else:
            # Si no tiene prefijo, es continuación del mensaje actual
            if current_message:
                # Buscar si ya hay contenido de texto
                text_content = None
                for content in current_message["content"]:
                    if content["type"] == "text":
                        text_content = content
                        break
                
                if text_content:
                    # Agregar la línea al texto existente
                    text_content["text"] = f"{text_content['text']}\n{line}"
                else:
                    # Agregar nuevo contenido de texto
                    current_message["content"].append({"type": "text", "text": line})
            else:
                # Si no hay mensaje actual, asumir que es mensaje de usuario
                current_message = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": line}
                    ]
                }
                current_role = "user"
    
    # Agregar el último mensaje si existe
    if current_message:
        messages.append(current_message)
    
    # Asegurar que los roles alternen correctamente
    cleaned_messages = []
    last_role = None
    
    for message in messages:
        current_role = message["role"]
        
        # Si es el primer mensaje, agregarlo
        if last_role is None:
            cleaned_messages.append(message)
            last_role = current_role
            continue
        
        # Si el rol actual es diferente al anterior, agregarlo
        if current_role != last_role:
            cleaned_messages.append(message)
            last_role = current_role
        else:
            # Si hay roles consecutivos iguales, combinar el contenido
            if cleaned_messages:
                # Combinar el contenido del mensaje actual con el último mensaje del mismo rol
                last_message = cleaned_messages[-1]
                
                # Agregar todo el contenido del mensaje actual al último mensaje
                for content in message["content"]:
                    if content["type"] == "text":
                        # Buscar si ya existe contenido de texto
                        existing_text = None
                        for existing_content in last_message["content"]:
                            if existing_content["type"] == "text":
                                existing_text = existing_content
                                break
                        
                        if existing_text:
                            existing_text["text"] = f"{existing_text['text']}\n{content['text']}"
                        else:
                            last_message["content"].append(content)
                    else:
                        # Para imágenes, agregar directamente
                        last_message["content"].append(content)
    
    print("*****************************************cleaned_messages_with_images****************************************")
    print(cleaned_messages)
    
    return cleaned_messages