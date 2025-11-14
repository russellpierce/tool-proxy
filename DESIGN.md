# LLM Direct Communication Proxy - Design Document

## Overview

This project implements a system where LLMs can communicate directly with each other without requiring tool calls. The system acts as a proxy that intercepts LLM requests and responses, allowing custom logic to be injected at various points in the request/response lifecycle.

### Primary Use Case

Enable transparent LLM-to-LLM communication where:
1. A user makes a request to one LLM provider (e.g., a local model)
2. The proxy intercepts the response
3. The proxy can call a different LLM provider (e.g., OpenAI) to analyze/enhance/modify the response
4. The final result is returned to the user

This enables patterns like:
- Using cheap local models for initial processing, then calling premium models for refinement
- Content validation/safety filtering via a specialized model
- Response enhancement or fact-checking
- Multi-model consensus systems

## Architecture Decision

After research into LiteLLM's capabilities, we've chosen to build this system using **LiteLLM Proxy Mode with Custom Logger Plugins**.

### Why LiteLLM?

LiteLLM provides:
- **Unified API**: Single interface for 1.8k+ models across 80+ providers
- **OpenAI-Compatible**: Works as a drop-in replacement for OpenAI API
- **Plugin System**: CustomLogger hooks for request/response interception
- **Provider Abstraction**: Automatic handling of provider-specific formats
- **Built-in Features**: Cost tracking, load balancing, retries, fallbacks

### Why CustomLogger Approach?

LiteLLM offers two extension mechanisms:

1. **Custom Providers** (`BaseCustomProvider`): For adding new LLM backends
2. **Custom Loggers** (`BaseCustomLogger`): For intercepting/modifying requests and responses

We're using **CustomLogger** because:
- We don't need to add new LLM providers (LiteLLM already supports the providers we need)
- We need to intercept and modify responses from existing providers
- We need to make secondary LLM calls during the request lifecycle
- We want observability and logging capabilities

## Key Design Decisions

### 1. Non-Streaming Focus

**Decision**: Focus on non-streaming responses for modification.

**Rationale**:
- LiteLLM's `async_post_call_success_hook` works reliably for non-streaming responses
- The `async_post_call_streaming_iterator_hook` has known issues in proxy mode (Issue #9639)
- Our primary use case doesn't require real-time streaming modification
- Streaming responses can still be observed/logged, just not modified

**Trade-off**: Users requesting streaming will get streaming responses, but our modifications won't apply. For use cases requiring modification, the client should request non-streaming.

### 2. Cross-Provider Communication Pattern

**Decision**: Use `litellm.acompletion()` with `no-log=True` for secondary LLM calls.

**Rationale**:
- Prevents infinite callback recursion
- Reuses LiteLLM's provider abstraction
- Maintains type safety and error handling
- Enables calling any provider from within the plugin

**Implementation Pattern**:
```python
class CrossProviderPlugin(BaseCustomLogger):
    async def async_post_call_success_hook(self, data, user_api_key_dict, response):
        # Primary request completed (e.g., from local model)

        # Make secondary call to different provider
        analysis = await litellm.acompletion(
            model="openai/gpt-4",  # Different provider!
            messages=[{
                "role": "user",
                "content": f"Analyze: {response.choices[0].message.content}"
            }],
            **{"no-log": True}  # Critical: prevents callback recursion
        )

        # Combine results, modify response, etc.
        return response
```

### 3. Provider Specification Format

**Decision**: Use explicit `provider/model` format for clarity.

**Rationale**:
- LiteLLM supports both implicit (`"gpt-4"`) and explicit (`"openai/gpt-4"`) formats
- Explicit format prevents ambiguity when multiple providers could serve the same model
- More maintainable and self-documenting
- Required for some providers (Azure, Ollama, VLLM, etc.)

**Examples**:
```python
# Explicit provider specification (recommended)
model="openai/gpt-4"              # OpenAI
model="anthropic/claude-3-5-sonnet"  # Anthropic
model="ollama/llama2"             # Local Ollama
model="azure/my-deployment"       # Azure OpenAI
model="bedrock/anthropic.claude-v2"  # AWS Bedrock
```

### 4. Plugin Architecture

**Decision**: Inherit from `BaseCustomLogger` and implement specific hooks.

**Available Hooks**:

| Hook | Purpose | When It Runs | Can Modify |
|------|---------|--------------|------------|
| `async_pre_call_hook` | Modify requests before sending to LLM | Before LLM call | Request data |
| `async_post_call_success_hook` | Modify responses before returning to user | After LLM call (non-streaming) | Response object |
| `async_log_success_event` | Log successful completions | After LLM call | N/A (observability only) |
| `async_log_failure_event` | Log failed completions | After LLM error | N/A (observability only) |
| `async_post_call_streaming_hook` | Observe streaming responses | During streaming | N/A (broken in proxy mode) |

**Implementation Structure**:
```python
from litellm_plugin.logger import BaseCustomLogger
import litellm

class MyPlugin(BaseCustomLogger):
    async def async_post_call_success_hook(self, data, user_api_key_dict, response):
        # Your custom logic here
        # Can call other LLMs, modify response, etc.
        return response
```

## Implementation Status

### What's Been Built

The skeleton implementation provides:

1. **Base Classes**:
   - `BaseCustomProvider`: For custom LLM providers (not our primary use case)
   - `BaseCustomLogger`: For request/response interception (our primary use case)

2. **Plugin Registry**:
   - Registration system for custom providers
   - Automatic initialization with LiteLLM

3. **Example Implementations**:
   - `EchoProvider`: Simple custom provider example
   - `MockAPIProvider`: Mock provider for testing
   - `ResponseModifier`: Response modification example
   - `ContentFilter`: Content filtering example
   - `RequestLogger`: Observability example

4. **Documentation**:
   - Comprehensive README with examples
   - Type hints throughout
   - Usage examples in `example_usage.py`

### What Needs to Be Built

Based on the design requirements:

1. **Cross-Provider Communication Plugin**:
   - Implement a CustomLogger that demonstrates calling a different provider
   - Example: Local model → OpenAI analysis pattern
   - Handle errors gracefully
   - Respect rate limits and costs

2. **Configuration Management**:
   - Environment variable handling for API keys
   - Provider-specific configuration
   - Model selection logic

3. **Testing**:
   - Unit tests for plugins
   - Integration tests with mock providers
   - Cross-provider communication tests

4. **Deployment Configuration**:
   - LiteLLM proxy server config
   - Docker/container configuration
   - Production deployment guide

## Technical Considerations

### Preventing Callback Recursion

**Problem**: If a CustomLogger calls `litellm.completion()`, it will trigger the same callbacks, causing infinite recursion.

**Solution**: Use `no-log=True` parameter:
```python
# This WILL cause infinite recursion:
response = await litellm.acompletion(model="gpt-4", messages=[...])

# This WON'T trigger callbacks:
response = await litellm.acompletion(
    model="gpt-4",
    messages=[...],
    **{"no-log": True}  # Disables all callbacks
)
```

### Error Handling

CustomLogger hooks should handle errors gracefully:

```python
async def async_post_call_success_hook(self, data, user_api_key_dict, response):
    try:
        # Secondary LLM call
        analysis = await litellm.acompletion(
            model="openai/gpt-4",
            messages=[...],
            **{"no-log": True}
        )
    except Exception as e:
        # Log error but don't fail the original request
        print(f"Secondary LLM call failed: {e}")
        # Return original response unmodified
        return response

    # Modify response based on analysis
    return response
```

### Performance Implications

**Latency**:
- Each secondary LLM call adds latency to the response
- Consider using faster models for secondary calls
- Implement timeouts to prevent hanging requests

**Cost**:
- Every secondary call incurs additional API costs
- Track costs using LiteLLM's built-in cost tracking
- Consider implementing usage limits

**Rate Limits**:
- Secondary calls count toward provider rate limits
- Implement exponential backoff for retries
- Consider caching results when appropriate

### Security Considerations

**API Keys**:
- Store API keys securely (environment variables, secrets management)
- Never log API keys
- Use separate keys for different providers when possible

**Content Safety**:
- Validate/sanitize content before sending to secondary models
- Implement PII detection/redaction if needed
- Consider using content filtering models

**Access Control**:
- LiteLLM proxy supports authentication via API keys
- Implement user-level usage tracking
- Rate limit per user/tenant

## Future Enhancements

### 1. Streaming Support

When LiteLLM fixes Issue #9639, we can enable streaming response modification:
- Real-time content filtering
- Progressive enhancement of streaming responses
- Multi-model streaming consensus

### 2. Advanced Routing

Implement intelligent model routing:
- Cost-based routing (try cheap model first, escalate if needed)
- Capability-based routing (use specialized models for specific tasks)
- Geographic routing (minimize latency)

### 3. Caching Layer

Add response caching:
- Cache secondary LLM analysis results
- Semantic similarity matching for cache hits
- TTL-based cache invalidation

### 4. Observability

Enhanced monitoring:
- Distributed tracing across LLM calls
- Cost analytics per user/model
- Performance metrics and alerting
- A/B testing framework

### 5. Multi-Agent Patterns

Support complex multi-agent workflows:
- Consensus voting across multiple models
- Specialist model orchestration
- Iterative refinement loops

## Configuration Examples

### LiteLLM Proxy Config

```yaml
model_list:
  # Primary models (user-facing)
  - model_name: local-llama
    litellm_params:
      model: ollama/llama2
      api_base: http://localhost:11434

  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: os.environ/OPENAI_API_KEY

  - model_name: claude
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY

litellm_settings:
  # Register custom logger
  callbacks: ["litellm_plugin.cross_provider.CrossProviderPlugin"]

  # Other settings
  drop_params: true
  set_verbose: false
```

### Environment Variables

```bash
# Provider API Keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export AZURE_API_KEY="..."

# Plugin Configuration
export SECONDARY_MODEL="openai/gpt-4"
export ENABLE_ANALYSIS="true"
export MAX_SECONDARY_CALLS="100"  # Rate limiting
```

## Comparison with Alternatives

### Why Not Build a Custom Proxy?

**Considered**: Building a FastAPI/Flask proxy from scratch

**Rejected because**:
- LiteLLM already handles provider integrations
- Would need to reimplement retry logic, rate limiting, cost tracking
- More maintenance burden
- LiteLLM is battle-tested and actively maintained

### Why Not Use LiteLLM SDK Directly?

**Considered**: Using LiteLLM as a library in our own service

**Rejected because**:
- LiteLLM Proxy provides OpenAI-compatible API out of the box
- Built-in authentication, load balancing, caching
- CustomLogger hooks work in both SDK and Proxy modes
- Easier deployment and scaling

### Why Not Use Other LLM Gateways?

**Alternatives considered**:
- Portkey
- Kong with AI Gateway plugin
- Custom Nginx/Envoy setup

**LiteLLM chosen because**:
- Open source and self-hostable
- Most comprehensive provider support
- Active development and community
- Explicit plugin/callback system
- Python-native (easier for ML/AI teams)

## References

### LiteLLM Documentation

- [Custom Callbacks](https://docs.litellm.ai/docs/observability/custom_callback)
- [Modify/Reject Incoming Requests](https://docs.litellm.ai/docs/proxy/call_hooks)
- [Provider Documentation](https://docs.litellm.ai/docs/providers)
- [Proxy Configuration](https://docs.litellm.ai/docs/proxy/configs)

### Known Issues

- [Issue #9639: Cannot Modify Streaming Responses](https://github.com/BerriAI/litellm/issues/9639)
  - `async_post_call_streaming_iterator_hook` not invoked in proxy mode
  - Affects streaming response modification only
  - Non-streaming modification works correctly

### Related Projects

- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [LiteLLM Proxy](https://docs.litellm.ai/docs/simple_proxy)

## Conclusion

This design leverages LiteLLM's robust proxy infrastructure and plugin system to enable transparent LLM-to-LLM communication. The CustomLogger approach provides the flexibility to intercept and modify requests/responses while maintaining compatibility with LiteLLM's extensive provider ecosystem.

The architecture is:
- **Simple**: Inherits from BaseCustomLogger, implements hooks
- **Powerful**: Can call any LLM provider from within hooks
- **Safe**: `no-log=True` prevents callback recursion
- **Production-Ready**: Built on battle-tested LiteLLM infrastructure
- **Extensible**: Easy to add new patterns and behaviors

Next steps involve implementing specific cross-provider communication patterns and deploying the proxy with appropriate security and monitoring.
