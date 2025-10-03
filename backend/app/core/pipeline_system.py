# prompt_to_pipeline/core/pipeline_system.py

import json
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import re
from datetime import datetime
import uuid

# ================== Data Models ==================

class TaskType(Enum):
    SUMMARIZATION = "summarization"
    QUIZ_GENERATION = "quiz_generation"
    PRESENTATION = "presentation"
    DATA_ANALYSIS = "data_analysis"
    CODE_GENERATION = "code_generation"
    TRANSLATION = "translation"
    EXTRACTION = "extraction"
    CUSTOM = "custom"

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Task:
    id: str
    name: str
    type: TaskType
    description: str
    input_data: Any
    output_data: Optional[Any] = None
    dependencies: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    error_message: Optional[str] = None
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type.value,
            'description': self.description,
            'status': self.status.value,
            'dependencies': self.dependencies,
            'params': self.params
        }

@dataclass
class Pipeline:
    id: str
    name: str
    tasks: List[Task]
    execution_order: List[str]
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_task(self, task_id: str) -> Optional[Task]:
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'tasks': [task.to_dict() for task in self.tasks],
            'execution_order': self.execution_order,
            'created_at': self.created_at.isoformat()
        }

# ================== Intent Parser ==================

class IntentParser:
    def __init__(self):
        self.intent_patterns = {
            TaskType.SUMMARIZATION: [
                r'\bsummarize\b', r'\bsummary\b', r'\bcondense\b', 
                r'\bbrief\b', r'\bdigest\b', r'\btldr\b'
            ],
            TaskType.QUIZ_GENERATION: [
                r'\bquiz\b', r'\bquestions?\b', r'\btest\b', 
                r'\bassessment\b', r'\bexam\b', r'\bq&a\b'
            ],
            TaskType.PRESENTATION: [
                r'\bslides?\b', r'\bpowerpoint\b', r'\bppt\b', 
                r'\bpresentation\b', r'\bdeck\b'
            ],
            TaskType.DATA_ANALYSIS: [
                r'\banalyze\b', r'\banalysis\b', r'\bstatistics\b', 
                r'\binsights?\b', r'\btrends?\b', r'\bmetrics\b'
            ],
            TaskType.CODE_GENERATION: [
                r'\bcode\b', r'\bimplement\b', r'\bprogram\b', 
                r'\bscript\b', r'\bfunction\b', r'\balgorithm\b'
            ],
            TaskType.TRANSLATION: [
                r'\btranslate\b', r'\bconvert\s+to\b', r'\bin\s+\w+\s+language\b'
            ],
            TaskType.EXTRACTION: [
                r'\bextract\b', r'\bpull\s+out\b', r'\bfind\b', 
                r'\bget\b', r'\bretrieve\b'
            ]
        }
    
    def parse(self, user_input: str) -> List[TaskType]:
        """Parse user input to identify intended task types"""
        user_input_lower = user_input.lower()
        detected_intents = []
        
        for task_type, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, user_input_lower):
                    if task_type not in detected_intents:
                        detected_intents.append(task_type)
                    break
        
        # If no specific intent detected, classify as custom
        if not detected_intents:
            detected_intents.append(TaskType.CUSTOM)
        
        return detected_intents

# ================== Task Decomposer ==================

class TaskDecomposer:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.task_templates = {
            TaskType.SUMMARIZATION: {
                'name': 'Summarization',
                'description': 'Condense content into key points',
                'default_params': {'length': 'medium', 'style': 'bullet_points'}
            },
            TaskType.QUIZ_GENERATION: {
                'name': 'Quiz Generation',
                'description': 'Create questions based on content',
                'default_params': {'num_questions': 5, 'difficulty': 'medium'}
            },
            TaskType.PRESENTATION: {
                'name': 'Presentation Creation',
                'description': 'Build slide deck from content',
                'default_params': {'num_slides': 10, 'template': 'professional'}
            }
        }
    
    def decompose(self, user_input: str, detected_intents: List[TaskType]) -> List[Task]:
        """Decompose user request into executable tasks"""
        tasks = []
        
        # Analyze the input for task sequencing
        task_sequence = self._determine_sequence(user_input, detected_intents)
        
        # Create tasks based on the sequence
        for i, task_type in enumerate(task_sequence):
            task_id = f"task_{uuid.uuid4().hex[:8]}"
            
            # Determine dependencies
            dependencies = []
            if i > 0:
                dependencies.append(tasks[i-1].id)
            
            task = Task(
                id=task_id,
                name=self.task_templates.get(task_type, {}).get('name', task_type.value),
                type=task_type,
                description=self.task_templates.get(task_type, {}).get('description', ''),
                input_data=None,  # Will be resolved during execution
                dependencies=dependencies,
                params=self.task_templates.get(task_type, {}).get('default_params', {})
            )
            
            tasks.append(task)
        
        return tasks
    
    def _determine_sequence(self, user_input: str, intents: List[TaskType]) -> List[TaskType]:
        """Determine the logical sequence of tasks"""
        # Simple heuristic: certain tasks naturally follow others
        sequence = []
        
        # Check for explicit ordering in the input
        if "then" in user_input.lower() or "and" in user_input.lower():
            # Parse sequential instructions
            parts = re.split(r'\b(?:then|and|after that|followed by)\b', user_input.lower())
            
            for part in parts:
                for intent in intents:
                    patterns = self._get_patterns_for_type(intent)
                    for pattern in patterns:
                        if re.search(pattern, part):
                            if intent not in sequence:
                                sequence.append(intent)
                            break
        
        # If no explicit sequence, use logical ordering
        if not sequence:
            sequence = self._apply_logical_ordering(intents)
        
        return sequence
    
    def _get_patterns_for_type(self, task_type: TaskType) -> List[str]:
        """Get regex patterns for a task type"""
        parser = IntentParser()
        return parser.intent_patterns.get(task_type, [])
    
    def _apply_logical_ordering(self, intents: List[TaskType]) -> List[TaskType]:
        """Apply logical task ordering rules"""
        priority_order = [
            TaskType.EXTRACTION,
            TaskType.TRANSLATION,
            TaskType.SUMMARIZATION,
            TaskType.DATA_ANALYSIS,
            TaskType.QUIZ_GENERATION,
            TaskType.CODE_GENERATION,
            TaskType.PRESENTATION,
            TaskType.CUSTOM
        ]
        
        return sorted(intents, key=lambda x: priority_order.index(x) if x in priority_order else 999)

# ================== Task Modules ==================

class TaskModule(ABC):
    """Abstract base class for task modules"""
    
    @abstractmethod
    async def execute(self, input_data: Any, params: Dict[str, Any]) -> Any:
        pass

class SummarizationModule(TaskModule):
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
    
    async def execute(self, input_data: Any, params: Dict[str, Any]) -> str:
        """Execute summarization task"""
        # Simulate LLM call with placeholder
        if isinstance(input_data, dict):
            text = input_data.get('text', str(input_data))
        else:
            text = str(input_data)
        
        length = params.get('length', 'medium')
        style = params.get('style', 'paragraph')
        
        # Placeholder for actual LLM summarization
        summary = f"[Summary of {len(text)} characters in {length} length, {style} style]:\n"
        summary += f"• Key point 1 from the content\n"
        summary += f"• Key point 2 from the content\n"
        summary += f"• Key point 3 from the content"
        
        await asyncio.sleep(0.5)  # Simulate processing time
        return summary

class QuizGeneratorModule(TaskModule):
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
    
    async def execute(self, input_data: Any, params: Dict[str, Any]) -> Dict:
        """Generate quiz questions from content"""
        num_questions = params.get('num_questions', 5)
        difficulty = params.get('difficulty', 'medium')
        
        # Placeholder quiz generation
        quiz = {
            'title': 'Generated Quiz',
            'questions': []
        }
        
        for i in range(num_questions):
            quiz['questions'].append({
                'id': f'q{i+1}',
                'question': f'Sample question {i+1} ({difficulty} difficulty)',
                'options': ['Option A', 'Option B', 'Option C', 'Option D'],
                'correct_answer': 'A',
                'explanation': f'Explanation for question {i+1}'
            })
        
        await asyncio.sleep(0.5)
        return quiz

class PresentationModule(TaskModule):
    def __init__(self):
        pass
    
    async def execute(self, input_data: Any, params: Dict[str, Any]) -> Dict:
        """Create presentation slides from content"""
        num_slides = params.get('num_slides', 10)
        template = params.get('template', 'professional')
        
        # Extract content for slides
        content = str(input_data)
        
        presentation = {
            'title': 'Generated Presentation',
            'template': template,
            'slides': []
        }
        
        # Title slide
        presentation['slides'].append({
            'type': 'title',
            'title': 'Presentation Title',
            'subtitle': 'Auto-generated from content'
        })
        
        # Content slides
        for i in range(min(num_slides - 2, 5)):  # Cap at 5 content slides for demo
            presentation['slides'].append({
                'type': 'content',
                'title': f'Section {i+1}',
                'content': f'Content for slide {i+1} based on input data',
                'bullet_points': [
                    f'Point {j+1} for section {i+1}' 
                    for j in range(3)
                ]
            })
        
        # Summary slide
        presentation['slides'].append({
            'type': 'summary',
            'title': 'Summary',
            'content': 'Key takeaways from the presentation'
        })
        
        await asyncio.sleep(0.7)
        return presentation

# ================== Pipeline Builder ==================

class PipelineBuilder:
    def __init__(self):
        pass
    
    def build(self, tasks: List[Task], user_input: str) -> Pipeline:
        """Build an executable pipeline from tasks"""
        # Determine execution order based on dependencies
        execution_order = self._topological_sort(tasks)
        
        pipeline = Pipeline(
            id=f"pipeline_{uuid.uuid4().hex[:12]}",
            name=f"Pipeline for: {user_input[:50]}...",
            tasks=tasks,
            execution_order=execution_order,
            created_at=datetime.now(),
            metadata={'user_input': user_input}
        )
        
        return pipeline
    
    def _topological_sort(self, tasks: List[Task]) -> List[str]:
        """Perform topological sort to determine execution order"""
        # Build adjacency list
        graph = {task.id: task.dependencies for task in tasks}
        
        # Calculate in-degree
        in_degree = {task.id: 0 for task in tasks}
        for task in tasks:
            for dep in task.dependencies:
                if dep in in_degree:
                    in_degree[dep] += 1
        
        # Find all nodes with 0 in-degree
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        execution_order = []
        
        while queue:
            task_id = queue.pop(0)
            execution_order.append(task_id)
            
            # Reduce in-degree for dependent tasks
            for task in tasks:
                if task_id in task.dependencies:
                    in_degree[task.id] -= 1
                    if in_degree[task.id] == 0:
                        queue.append(task.id)
        
        return execution_order

# ================== Pipeline Executor ==================

class PipelineExecutor:
    def __init__(self):
        self.modules = {
            TaskType.SUMMARIZATION: SummarizationModule(),
            TaskType.QUIZ_GENERATION: QuizGeneratorModule(),
            TaskType.PRESENTATION: PresentationModule(),
        }
        self.execution_history = []
    
    async def execute_pipeline(self, pipeline: Pipeline, input_data: Any = None) -> Dict[str, Any]:
        """Execute all tasks in the pipeline"""
        results = {}
        pipeline_input = input_data or pipeline.metadata.get('user_input', '')
        
        for task_id in pipeline.execution_order:
            task = pipeline.get_task(task_id)
            if not task:
                continue
            
            try:
                # Update task status
                task.status = TaskStatus.RUNNING
                
                # Resolve input from dependencies
                if task.dependencies:
                    # Use output from the last dependency
                    last_dep = task.dependencies[-1]
                    task_input = results.get(last_dep, pipeline_input)
                else:
                    task_input = pipeline_input
                
                # Execute the task
                module = self.modules.get(task.type)
                if module:
                    print(f"Executing: {task.name} ({task.type.value})")
                    result = await module.execute(task_input, task.params)
                    task.output_data = result
                    results[task_id] = result
                    task.status = TaskStatus.COMPLETED
                else:
                    raise ValueError(f"No module found for task type: {task.type}")
                
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error_message = str(e)
                print(f"Task {task.name} failed: {e}")
        
        # Record execution
        self.execution_history.append({
            'pipeline_id': pipeline.id,
            'timestamp': datetime.now(),
            'results': results
        })
        
        return {
            'pipeline_id': pipeline.id,
            'status': 'completed',
            'results': results,
            'execution_order': pipeline.execution_order
        }

# ================== Main Orchestrator ==================

class PromptToPipeline:
    def __init__(self):
        self.intent_parser = IntentParser()
        self.task_decomposer = TaskDecomposer()
        self.pipeline_builder = PipelineBuilder()
        self.pipeline_executor = PipelineExecutor()
    
    async def process(self, user_input: str, input_data: Any = None) -> Dict[str, Any]:
        """Main entry point for processing user prompts"""
        print(f"\n{'='*50}")
        print(f"Processing: {user_input}")
        print(f"{'='*50}\n")
        
        # Step 1: Parse intent
        detected_intents = self.intent_parser.parse(user_input)
        print(f"Detected intents: {[intent.value for intent in detected_intents]}")
        
        # Step 2: Decompose into tasks
        tasks = self.task_decomposer.decompose(user_input, detected_intents)
        print(f"Created {len(tasks)} tasks")
        
        # Step 3: Build pipeline
        pipeline = self.pipeline_builder.build(tasks, user_input)
        print(f"Built pipeline with execution order: {pipeline.execution_order}")
        
        # Step 4: Execute pipeline
        results = await self.pipeline_executor.execute_pipeline(pipeline, input_data)
        
        print(f"\n{'='*50}")
        print(f"Pipeline completed successfully!")
        print(f"{'='*50}\n")
        
        return {
            'pipeline': pipeline.to_dict(),
            'results': results,
            'summary': self._generate_summary(pipeline, results)
        }
    
    def _generate_summary(self, pipeline: Pipeline, results: Dict) -> str:
        """Generate a summary of the pipeline execution"""
        completed_tasks = [t for t in pipeline.tasks if t.status == TaskStatus.COMPLETED]
        failed_tasks = [t for t in pipeline.tasks if t.status == TaskStatus.FAILED]
        
        summary = f"Pipeline executed {len(completed_tasks)}/{len(pipeline.tasks)} tasks successfully."
        if failed_tasks:
            summary += f" {len(failed_tasks)} task(s) failed."
        return summary

# ================== Example Usage ==================

async def main():
    # Initialize the system
    ptp = PromptToPipeline()
    
    # Example prompts
    test_prompts = [
        "Summarize this research paper and generate 5 quiz questions",
        "Extract key data, analyze trends, and create a presentation",
        "Translate this document to Spanish, then summarize it and create slides",
    ]
    
    for prompt in test_prompts:
        try:
            # Process the prompt
            result = await ptp.process(prompt, input_data="Sample input text for processing...")
            
            # Display results
            print(f"Pipeline ID: {result['pipeline']['id']}")
            print(f"Summary: {result['summary']}")
            print(f"Tasks executed: {len(result['pipeline']['tasks'])}")
            print("-" * 50)
            
        except Exception as e:
            print(f"Error processing prompt: {e}")
            print("-" * 50)

if __name__ == "__main__":
    # Run the example
    asyncio.run(main())