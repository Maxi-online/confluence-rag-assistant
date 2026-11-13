"""
Функции handoff.
Определяют передачу управления между агентами.
Возвращают экземпляры соответствующих агентов по user_profile_id.
"""

import inspect
from typing import Any, Dict
from ai.mas.agents.base_agent import Agent

def function_to_schema(func) -> dict:
    """
    Converts a Python function to an OpenAI function schema.
    
    Args:
        func: Python function to convert
        
    Returns:
        dict: OpenAI function schema
    """
    type_map = {
        str: "string",
        int: "integer", 
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )

    parameters = {}
    for param in signature.parameters.values():
        try:
            param_type = type_map.get(param.annotation, "string")
        except KeyError as e:
            raise KeyError(
                f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}"
            )
        parameters[param.name] = {"type": param_type}

    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty
    ]

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": (func.__doc__ or "").strip(),
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }

def transfer_to_language_detection_agent(
    knowledge_client: Any,
    user_profile_id: str,
    openai_api_key: str,
    openai_base_url: str = "https://api.openai.com/v1/",
    request: Any = None,
    **kwargs
) -> Agent:
    """
    Returns a LanguageDetectionAgent instance.
    Call this function to transfer control to the language detection specialist.
    
    Args:
        knowledge_client: Client for knowledge base interaction
        user_profile_id: User profile ID
        openai_api_key: OpenAI API key
        openai_base_url: Base URL for OpenAI API
        request: Additional request data
        **kwargs: Additional keyword arguments
        
    Returns:
        Agent: Language detection agent instance
    """
    from ai.mas.agents.language_detection_agent import LanguageDetectionAgent
    return LanguageDetectionAgent(
        knowledge_client=knowledge_client,
        user_profile_id=user_profile_id,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        request=request,
        **kwargs
    )

def transfer_to_common_questions_agent(
    knowledge_client: Any,
    user_profile_id: str,
    openai_api_key: str,
    openai_base_url: str = "https://api.openai.com/v1/",
    request: Any = None,
    **kwargs
) -> Agent:
    """
    Returns a CommonQuestionsAgent instance.
    Call this function when user has general questions about the university.
    
    Args:
        knowledge_client: Client for knowledge base interaction
        user_profile_id: User profile ID
        openai_api_key: OpenAI API key
        openai_base_url: Base URL for OpenAI API
        request: Additional request data
        **kwargs: Additional keyword arguments
        
    Returns:
        Agent: Common questions agent instance
    """
    from ai.mas.agents.common_questions_agent import CommonQuestionsAgent
    return CommonQuestionsAgent(
        knowledge_client=knowledge_client,
        user_profile_id=user_profile_id,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        request=request,
        **kwargs
    )

def transfer_to_triage_agent(
    knowledge_client: Any,
    user_profile_id: str,
    openai_api_key: str,
    openai_base_url: str = "https://api.openai.com/v1/",
    request: Any = None,
    **kwargs
) -> Agent:
    """
    Returns a TriageAgent instance.
    Call this function when you need to collect basic user information.
    
    Args:
        knowledge_client: Client for knowledge base interaction
        user_profile_id: User profile ID
        openai_api_key: OpenAI API key
        openai_base_url: Base URL for OpenAI API
        request: Additional request data
        **kwargs: Additional keyword arguments
        
    Returns:
        Agent: Triage agent instance
    """
    from ai.mas.agents.triage_agent import TriageAgent
    return TriageAgent(
        knowledge_client=knowledge_client,
        user_profile_id=user_profile_id,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        request=request,
        **kwargs
    )

def transfer_to_program_decision_agent(
    knowledge_client: Any,
    user_profile_id: str,
    openai_api_key: str,
    openai_base_url: str = "https://api.openai.com/v1/",
    request: Any = None,
    **kwargs
) -> Agent:
    """
    Returns a ProgramDecisionAgent instance.
    Call this function when user needs help selecting a program or course.
    
    Args:
        knowledge_client: Client for knowledge base interaction
        user_profile_id: User profile ID
        openai_api_key: OpenAI API key
        openai_base_url: Base URL for OpenAI API
        request: Additional request data
        **kwargs: Additional keyword arguments
        
    Returns:
        Agent: Program decision agent instance
    """
    from ai.mas.agents.program_decision_agent import ProgramDecisionAgent
    return ProgramDecisionAgent(
        knowledge_client=knowledge_client,
        user_profile_id=user_profile_id,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        request=request,
        **kwargs
    )

def transfer_to_authorization_agent(
    knowledge_client: Any,
    user_profile_id: str,
    openai_api_key: str,
    openai_base_url: str = "https://api.openai.com/v1/",
    request: Any = None,
    **kwargs
) -> Agent:
    """
    Returns an AuthorizationAgent instance.
    Call this function when user needs to register or authorize.
    
    Args:
        knowledge_client: Client for knowledge base interaction
        user_profile_id: User profile ID
        openai_api_key: OpenAI API key
        openai_base_url: Base URL for OpenAI API
        request: Additional request data
        **kwargs: Additional keyword arguments
        
    Returns:
        Agent: Authorization agent instance
    """
    from ai.mas.agents.authorization_agent import AuthorizationAgent
    return AuthorizationAgent(
        knowledge_client=knowledge_client,
        user_profile_id=user_profile_id,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        request=request,
        **kwargs
    )

def transfer_to_lead_qualification_agent(
    knowledge_client: Any,
    user_profile_id: str,
    openai_api_key: str,
    openai_base_url: str = "https://api.openai.com/v1/",
    request: Any = None,
    **kwargs
) -> Agent:
    """
    Returns a LeadQualificationAgent instance.
    Call this function when user is ready for detailed interview.
    
    Args:
        knowledge_client: Client for knowledge base interaction
        user_profile_id: User profile ID
        openai_api_key: OpenAI API key
        openai_base_url: Base URL for OpenAI API
        request: Additional request data
        **kwargs: Additional keyword arguments
        
    Returns:
        Agent: Lead qualification agent instance
    """
    from ai.mas.agents.lead_qualification_agent import LeadQualificationAgent
    return LeadQualificationAgent(
        knowledge_client=knowledge_client,
        user_profile_id=user_profile_id,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        request=request,
        **kwargs
    )

def transfer_to_objection_detector_agent(
    user_profile_id: str, openai_api_key: str, knowledge_client: Any
) -> Agent:
    """
    Returns an ObjectionDetectorAgent instance.
    Call this function when you need to identify user's concerns.
    
    Args:
        user_profile_id: User profile ID
        openai_api_key: OpenAI API key
        knowledge_client: Client for knowledge base interaction
        
    Returns:
        Agent: Objection detector agent instance
    """
    from ai.mas.agents.objection_detector_agent import ObjectionDetectorAgent
    return ObjectionDetectorAgent(user_profile_id=user_profile_id, openai_api_key=openai_api_key, knowledge_client=knowledge_client)

def transfer_back_to_orchestrator(user_profile_id: str) -> None:
    """
    Call this function when you need to return control to the Orchestrator.
    
    Args:
        user_profile_id: User profile ID
    """
    from ai.mas.agents.orchestrator_agent import OrchestratorAgent
    print(f"[Handoff] Возвращаю управление Orchestrator для {user_profile_id}")
    # Note: This function is a placeholder as we can't actually return an Orchestrator instance here
    # The orchestrator will handle this signal in a special way

def send_to_sales_department(data: dict) -> None:
    """
    Send collected data to sales department.
    This currently just logs the data.

    Args:
        data: Dictionary with lead information
    """
    print(f"[Sales Department] Получены данные для обработки: {data}")
