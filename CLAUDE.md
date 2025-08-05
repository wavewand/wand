# Wand MCP Tool - Claude Context Guide

## IMPORTANT: Initial Setup Check
**Every Claude instance should perform these steps before proceeding:**

1. **Verify Wand MCP Server Connection**:
   ```bash
   claude mcp list
   ```
   Expected output should show: `wand: ... - ✓ Connected`

2. **Test Ollama Integration**:
   ```
   Use mcp__wand__ollama with operation "list_models"
   ```
   This confirms the Ollama server at `http://localhost:11434` is accessible (configure via OLLAMA_BASE_URL env var)

3. **If tools are not available**:
   - Run the add_to_claude.sh script from the wand directory
   - Restart Claude Desktop to load the MCP configuration
   - Verify connection again with steps 1 and 2

## Overview
Wand is a Model Context Protocol (MCP) server that provides 50+ integrations for AI development workflows. This guide explains how to use Wand's Ollama integration to offload heavy AI tasks to local models while maintaining quality control.

## Quick Reference for M1 Max 64GB Setup

```python
# Your optimal settings for most tasks
{
    "model": "qwen3:30b-a3b-instruct-2507-q4_K_M",  # Best quality
    "max_tokens": 4096,   # Can go up to 8192+ comfortably
    "temperature": 0.2    # Consistency over creativity
}

# For speed-critical simple tasks
{
    "model": "qwen2.5-coder:14b-instruct",
    "max_tokens": 2048,
    "temperature": 0.3
}
```

**Key Insight**: With 64GB RAM, prefer single large requests over decomposition. You can generate complete implementations without hitting memory limits.

## Available Wand Tools
When Wand is properly configured, you'll have access to MCP tools prefixed with `mcp__wand__`. The primary tool for AI code generation is:

### `mcp__wand__ollama`
Local AI model execution through Ollama server (default: `http://localhost:11434`, configure via OLLAMA_BASE_URL)

## Core Operations

### 1. Code Generation (`generate` operation)
Use this for heavy code generation tasks:
```
Tool: mcp__wand__ollama
Operation: generate
Parameters:
  - prompt: The code generation request
  - model: (optional) defaults to "qwen3:30b-a3b-instruct-2507-q4_K_M"
  - temperature: (optional) defaults to 0.7
  - max_tokens: (optional) defaults to 2048
```

### 2. Chat Completion (`chat` operation)
For conversational code assistance:
```
Tool: mcp__wand__ollama
Operation: chat
Parameters:
  - messages: Array of message objects [{role: "user/assistant", content: "..."}]
  - model: (optional) defaults to "qwen3:30b-a3b-instruct-2507-q4_K_M"
```

### 3. List Available Models (`list_models` operation)
Check which models are available:
```
Tool: mcp__wand__ollama
Operation: list_models
```

## Usage Guidelines

### CRITICAL: Output Limitations
⚠️ **IMPORTANT: The Wand Ollama tool has strict output limitations:**
- **Can only generate ONE file at a time**
- **May only generate partial files for large requests**
- **Output is limited by token constraints**

### Breaking Down Large Tasks
**ALWAYS break large coding tasks into small, focused requests:**

1. **Instead of:** "Generate a complete authentication system"
   **Do:**
   - Step 1: "Generate the User model with SQLAlchemy"
   - Step 2: "Generate the password hashing utilities"
   - Step 3: "Generate the JWT token functions"
   - Step 4: "Generate the registration endpoint"
   - Step 5: "Generate the login endpoint"

**For complex multi-component systems (e.g., real-time chat, task queues):**
1. **CRITICAL: Define interfaces first** - Write exact method signatures in a document
2. Generate core data models first
3. Generate individual manager classes separately
4. Specify exact method signatures components will use
5. Generate integration/broadcaster classes last
6. Review each component before proceeding to next

2. **For multi-file projects:**
   - Request each file/component separately
   - Start with core models/interfaces
   - Then generate dependent components
   - Finally generate integration code

3. **For large single files:**
   - Generate the class/function structure first
   - Then generate individual methods
   - Add error handling separately
   - Add validation logic as a separate step

### When to Use Wand's Ollama Integration

**ALWAYS use Wand for:**
1. **Individual functions or methods** (20-100 lines)
2. **Single component generation** (one class, one module)
3. **Boilerplate code generation** (single test file, one interface)
4. **Code refactoring** of specific functions
5. **Initial skeleton implementations** that need expansion
6. **Performance-intensive generation** that would slow down the main Claude session

**DO NOT use Wand for:**
1. Critical security-sensitive code
2. Complex business logic requiring deep context understanding
3. Very small code snippets (<20 lines)
4. Code requiring specific Claude knowledge or context
5. **Multi-file generation in a single request**
6. **Complete applications or systems in one prompt**

### Quality Control Process

**ALWAYS follow this workflow:**

1. **Prepare the prompt** - Be specific and include:
   - Language and framework requirements
   - Code style conventions from the project
   - Input/output specifications
   - Error handling requirements

2. **Execute via Wand**:
   ```
   Use mcp__wand__ollama with operation "generate" or "chat"
   Include clear, detailed prompts
   Specify constraints and requirements
   ```

3. **MANDATORY CODE REVIEW** - Before saving ANY generated code:
   - Review for API mismatches (wrong function names, parameters)
   - Check for incorrect type usage or undefined types
   - Verify proper error handling and return codes
   - Ensure logging uses correct log levels
   - Validate thread safety and synchronization
   - Check for memory leaks or uninitialized variables
   - Verify compatibility with target platform/framework

4. **Fix identified issues**:
   - Correct any API mismatches with actual implementations
   - Fix type definitions and forward declarations
   - Adjust error handling to match project patterns
   - Update logging to use project's log configuration
   - Fix synchronization primitives (mutexes, semaphores)
   - Ensure proper resource cleanup

5. **Validate the corrected code**:
   - Check syntax correctness
   - Verify it follows project conventions
   - Ensure security best practices
   - Validate logic and edge cases

6. **Save and Integration**:
   - Save corrected code to appropriate files
   - Test with existing codebase
   - Run linting and type checking
   - Verify all tests pass

## Example Workflows

### Example 1: Generate a REST API endpoint (BROKEN DOWN)
```python
# WRONG WAY - Too large for single request:
# "Generate complete FastAPI authentication system"

# RIGHT WAY - Break into smaller tasks:

# Task 1: Generate User model
prompt1 = """
Generate a SQLAlchemy User model for FastAPI with:
- id (primary key)
- email (unique, indexed)
- hashed_password
- is_active (boolean)
- created_at timestamp
"""

# Task 2: Generate password utilities
prompt2 = """
Generate password hashing functions using bcrypt:
- hash_password function
- verify_password function
- Use passlib.context.CryptContext
"""

# Task 3: Generate registration endpoint only
prompt3 = """
Generate a FastAPI POST /register endpoint:
- Accept email and password
- Use Pydantic model for validation
- Return success response
- Handle duplicate email error
"""
```

### Example 2: Generate unit tests (FOCUSED)
```python
# WRONG WAY - Too broad:
# "Generate all tests for entire UserService class"

# RIGHT WAY - Request specific test methods:
prompt = """
Generate pytest test for UserService.create_user method:
- Test successful user creation
- Test duplicate email error
- Test invalid email format
- Use pytest fixtures for database
- Mock the database session
"""

# Then separately request other test methods
```

### Example 3: Refactor single function
```python
# WRONG WAY - Entire file refactor:
# "Refactor complete JavaScript file to TypeScript"

# RIGHT WAY - Function by function:
prompt = """
Convert this single JavaScript function to TypeScript:
function processOrder(order) {
  // ... existing code
}

Add proper type annotations for:
- order parameter
- return type
- internal variables
"""

# Process each function separately
```

## Common Issues and Solutions

### Known Code Generation Issues

1. **Mixed Async/Sync Patterns**:
   - Issue: FastAPI endpoints may mix sync Session with async functions
   - Solution: Always specify "use AsyncSession" and "async/await throughout"

2. **Conflicting Model Definitions**:
   - Issue: Pydantic and SQLAlchemy models with same name
   - Solution: Explicitly name models (e.g., "UserCreate for Pydantic, User for SQLAlchemy")

3. **Missing Imports**:
   - Issue: Generated code assumes libraries not specified
   - Solution: Always list ALL required imports explicitly in prompt

4. **Incomplete Features**:
   - Issue: Complex requirements may be partially implemented
   - Solution: Request one feature at a time, verify before proceeding

5. **Integration Mismatches**:
   - Issue: Generated components assume methods/interfaces that don't exist
   - Solution: Specify exact method signatures and integration points in prompts

6. **Singleton vs Instance Confusion**:
   - Issue: Classes that should be singletons created as new instances
   - Solution: Explicitly state "use as singleton" or "create global instance"

7. **Time-based Default Values**:
   - Issue: `datetime.now()` executes at import time instead of runtime
   - Solution: Specify "use func.now() for server defaults" in SQLAlchemy models

8. **Inconsistent Method Naming**:
   - Issue: Components can't integrate due to method name mismatches
   - Solution: Provide exact method names that other components will call

### Model Selection Strategy

Optimized for M1 Max 64GB with extensive headroom:

| Task Type | Recommended Model | Token Limit | Temperature | When to Use |
|-----------|-------------------|-------------|-------------|-------------|
| Simple functions | qwen2.5-coder:7b-instruct | 2048 | 0.2 | Speed critical, basic impl |
| Medium complexity | qwen2.5-coder:14b-instruct | 4096 | 0.3 | Balanced speed/quality |
| Complex logic | qwen3:30b-a3b-instruct-2507-q4_K_M | 8192 | 0.2 | Production code, quality critical |
| Large refactoring | qwen2.5-coder:32b-instruct | 6144 | 0.3 | Large codebases |
| Documentation | qwen3:30b-a3b-instruct-2507-q4_K_M | 4096 | 0.5 | Comprehensive docs |
| Distributed systems | qwen3:30b-a3b-instruct-2507-q4_K_M | 8192 | 0.1 | Consensus, CRDT, complex algorithms |
| Learning/Prototypes | qwen2.5-coder:14b-instruct | 2048 | 0.3 | Quick examples, education |

**Key Finding**: Qwen3 30B produces significantly higher quality code (better error handling, docs, patterns) but is 40-60% slower than qwen2.5-coder variants.

### Performance Expectations

- **Response Times**:
  - Small prompts (< 500 tokens): 8-15 seconds
  - Medium prompts (500-1000 tokens): 15-35 seconds
  - Large prompts (> 1000 tokens): 35-60 seconds

- **Output Quality by Size**:
  - < 50 lines: Usually complete and correct
  - 50-100 lines: May have minor issues
  - > 100 lines: Often incomplete, break into smaller tasks

- **Success Factors**:
  - Pre-defined interfaces: 90% integration success
  - No interface planning: 40% integration success
  - Single component requests: 85% quality
  - Multi-file requests: 30% quality

### Model Quality Comparison (from testing)

| Aspect | Qwen2.5-coder | Qwen3 30B |
|--------|---------------|-----------|
| Speed | ✅ 40-60% faster | ❌ Slower |
| Documentation | Basic | ✅ Comprehensive |
| Error Handling | Basic try/except | ✅ Production-ready |
| Code Completeness | Functional | ✅ Handles edge cases |
| Token Efficiency | ✅ Concise | ❌ Verbose, may hit limits |
| Best For | Prototypes, simple tasks | Production code, complex logic |

### Ultra-Complex System Generation Comparison

Based on testing with distributed systems (Redlock, CRDT, WebRTC, Raft):

| Component | Qwen2.5-coder:14b | Qwen3:30b |
|-----------|-------------------|-----------|
| **Distributed Lock (Redlock)** | Basic skeleton, placeholders | Full implementation with clock drift handling |
| **CRDT Text Sync** | Simple character ops, no GC | Complete RGA with vector clocks & tombstones |
| **WebRTC Signaling** | Skeleton with "complex task" comments | Full STUN/TURN, encryption, topology mgmt |
| **Raft Consensus** | Class structure with pass statements | Complete with persistence, state machine, networking |
| **Algorithm Correctness** | Often simplified/incorrect | Generally correct implementations |
| **Production Readiness** | Requires significant work | Near production-ready |
| **Learning Value** | ✅ Good for understanding structure | ❌ May be overwhelming |
| **Response Pattern** | "This is complex, here's a starting point" | Attempts full implementation |

**Critical Insight**: For distributed systems and complex algorithms, always use Qwen3:30b. The qwen2.5-coder models acknowledge complexity but provide only educational skeletons.

### Optimizing Generation with M1 Max 64GB

**With your setup, token limits are rarely a constraint.** You can generate complete implementations in single requests.

**Recommended Approach for Complex Systems**:

**Option 1 - Single Large Request (Preferred with 64GB)**:
```python
# Your M1 Max can handle this comfortably
"max_tokens": 8192,  # Complete implementations
"temperature": 0.1   # Ensure consistency
```

**Option 2 - Component Decomposition (When You Want More Control)**:

1. **Never request complete systems** - Break into atomic components:
   ```
   ❌ BAD: "Generate a complete Raft consensus implementation"
   ✅ GOOD: "Generate only the leader election logic for Raft"
   ```

2. **Request specific methods individually**:
   ```
   Step 1: "Generate the request_vote method for Raft with full error handling"
   Step 2: "Generate the append_entries method for Raft log replication"
   Step 3: "Generate the snapshot handling methods"
   ```

3. **Use incremental building approach**:
   - First: Generate type definitions and interfaces only
   - Second: Generate core state management
   - Third: Generate individual protocol handlers
   - Fourth: Generate helper/utility functions
   - Fifth: Generate integration logic

4. **Specify "code only" when appropriate**:
   ```
   "Generate ONLY the code for CRDT merge operation, no explanations or comments"
   ```

5. **Request partial implementations explicitly**:
   ```
   "Generate the first 3 methods of the WebRTC signaling server:
    negotiate_peer_connection, handle_ice_candidate, optimize_topology"
   ```

## Workflow for Best Results

### Pre-Generation Planning:
1. **Define all interfaces first** in a document
2. **Map component dependencies** before generating
3. **List exact method signatures** each component needs
4. **Specify singleton vs instance** patterns upfront

### During Generation:
1. **One component per request** - never multiple files
2. **Include integration details** in prompts
3. **Reference interface document** in each prompt
4. **Test generated code** before next component

## Best Practices

1. **Prompt Engineering**:
   - Include examples of desired output format
   - Specify coding standards and conventions
   - Mention required imports and dependencies
   - Be explicit about error handling needs
   - Specify async/sync preferences upfront
   - Define exact method signatures for integration points
   - State whether classes should be singletons or instances
   - Specify runtime vs import-time for default values
   - Include "use func.now() not datetime.now()" for SQLAlchemy

2. **Model Selection Based on Response Patterns**:
   - **Use Qwen2.5-coder when**:
     - You need quick prototypes or examples
     - The task is well-defined and simple
     - You want educational/learning code
     - Speed is more important than completeness
   - **Use Qwen3:30b when**:
     - Implementing distributed algorithms
     - Need production-ready error handling
     - Complex state management required
     - Algorithmic correctness is critical
   - **Warning signs you need Qwen3:30b**:
     - Task involves consensus, synchronization, or replication
     - Multiple failure modes need handling
     - Byzantine fault tolerance required
     - Complex networking protocols

2. **Validation Checklist**:
   - [ ] Syntax is correct
   - [ ] Follows project code style
   - [ ] Has proper error handling
   - [ ] Includes necessary imports
   - [ ] No security vulnerabilities
   - [ ] Efficient algorithms used
   - [ ] Properly documented
   - [ ] Async/sync patterns are consistent
   - [ ] All required features are implemented
   - [ ] Type hints are complete and correct

3. **Performance Optimization**:
   - Use Wand for initial generation of large code blocks
   - Batch similar generation tasks together
   - Cache commonly used patterns

4. **Security Considerations**:
   - Never send sensitive data (passwords, API keys) to Ollama
   - Always review generated authentication/authorization code
   - Validate all database queries for injection vulnerabilities
   - Check for proper input sanitization

## Configuration

The Ollama integration connects to: `http://localhost:11434` (configure via OLLAMA_BASE_URL)
Default model: `qwen3:30b-a3b-instruct-2507-q4_K_M`
Timeout: 12 hours (for large generation tasks)

### Hardware Specifications
- **Server**: M1 Max with 64GB RAM
- **Model RAM Usage**: Qwen3:30b Q4_K_M uses ~18-20GB
- **Available for Generation**: ~40GB RAM headroom

### Self-Hosted Advantages

With your M1 Max 64GB setup:

**Key Advantages**:
- **No API rate limits** - Generate continuously
- **No usage costs** - Only electricity
- **Complete privacy** - Code never leaves your network
- **Generous token limits** - Can use 8192+ tokens comfortably
- **Fast inference** - M1 Max provides excellent performance

**Optimal Settings for M1 Max 64GB**:
```python
# Recommended defaults for your setup
"max_tokens": 4096,     # Comfortable default with headroom
"temperature": 0.2,     # Lower for consistency
"num_ctx": 8192        # Extended context window
```

**Token Limits by Task Type**:
| Task Complexity | Recommended max_tokens | Your Setup Can Handle |
|----------------|------------------------|----------------------|
| Simple functions | 1024-2048 | ✅ Easily |
| Full classes | 2048-4096 | ✅ Easily |
| Complex systems | 4096-6144 | ✅ Comfortably |
| Complete implementations | 8192-12288 | ✅ With ~40GB free |

**Performance Optimization**:
1. **Use higher token limits freely** - You have the RAM
2. **Keep models loaded** - With 64GB, no need to unload frequently
3. **Run multiple models** - Can keep both Qwen3:30b and Qwen2.5-coder loaded
4. **Parallel requests** - M1 Max handles concurrent inference well

## Troubleshooting

If Wand tools are not available:
1. Check that Wand MCP server is running
2. Verify Claude Desktop has been restarted after adding Wand
3. Ensure the virtual environment is activated
4. Check logs in the `logs/` directory of your wand installation

## Model Selection Guide

Available models on the Ollama server:
- `qwen3:30b-a3b-instruct-2507-q4_K_M` - Best for complex code generation
- `codellama` - Optimized for code completion
- `mistral` - Good for general purpose tasks
- `nomic-embed-text` - For generating embeddings

Choose models based on task complexity and required quality.

## Effective Prompt Templates

### For Functions/Methods:
```
Generate a [language] function that:
- Function name: [name]
- Parameters: [list with types]
- Returns: [return type]
- Purpose: [what it does]
- Include: error handling, type hints, docstring
- Use: [specific libraries if needed]
```

### For Classes/Components:
```
Generate a [language] [class/component] with:
- Name: [name]
- Properties/State: [list]
- Methods: [list main methods only]
- Framework: [React/FastAPI/etc]
- Include all imports
- Use [sync/async] patterns
- Usage: [singleton/instance per request/global]
```

### For Complex Systems (Qwen3:30b - Avoiding Token Limits):
```
Generate ONLY the [specific component] for [system name]:
- Component: [e.g., "leader election module"]
- Methods: [list 2-3 specific methods max]
- Include: type definitions for this component
- Assume these types exist: [list any dependencies]
- Code only, no explanations
```

### For Distributed Algorithms (Qwen3:30b):
```
Generate the [specific algorithm phase] implementation:
- Algorithm: [e.g., "Raft consensus"]
- Phase: [e.g., "vote request handling only"]
- Include: error handling, logging
- Assume these exist: [list interfaces/types]
- Focus on: [specific aspect like "network partition handling"]
```

### For Integration Components:
```
Generate a [language] class that integrates with:
- Dependency 1: [class name] with methods [list exact signatures]
- Dependency 2: [class name] with methods [list exact signatures]
- This class should provide: [list methods with exact signatures]
- Include error handling for missing dependencies
```

### For Tests:
```
Generate a [test framework] test for [function/class name]:
- Test: [specific scenario]
- Mock: [what to mock]
- Assert: [expected behavior]
- Include fixtures and imports
```

## Remember

- **Always verify** generated code before using in production
- **Use Wand to accelerate**, not replace careful coding
- **Maintain security** awareness with all generated code
- **Test thoroughly** after integration
- **Document** any modifications made to generated code
- **Break down large tasks** into smaller, focused prompts
- **Review common issues** section when code doesn't work as expected

This tool is designed to enhance productivity while maintaining code quality and security standards.
