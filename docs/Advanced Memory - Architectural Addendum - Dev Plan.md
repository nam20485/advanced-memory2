# **Architectural Addendum: Integrating Gemini API Features**

## **Phase 6: AI-Powered Interactive Features (Optional)**

**Objective:** Enhance the application by integrating features powered by the Gemini API. This phase introduces an "AI-Powered Strategy Explainer" and an "Interactive Q\&A" function, transforming the application from a data-delivery system into an interactive, conversational analysis tool. The goal is to significantly boost user engagement and comprehension by making the dense technical information more accessible and explorable on demand.

### **1\. New Component: The Gemini Service (C\#)**

A new, dedicated service will be created to encapsulate all interactions with the Google Gemini API. This adheres to the Single Responsibility Principle, promoting a clean separation of concerns, centralizing API client logic, and making the component independently testable and reusable.

* **Project:** Create a new C\# class library project named MyAgent.GeminiClient. This library will have no dependencies on the main ASP.NET Core application, ensuring it can be used in other parts of the system in the future (e.g., in a background worker service).  
* **Dependencies:** Add the official Google Cloud Vertex AI client library for .NET via the NuGet package manager. The primary package is Google.Cloud.AIPlatform.V1.  
* **Implementation:**  
  * Create a GeminiService class. This service will be registered as a Singleton in the main ApiService's dependency injection container, as the Gemini client is thread-safe and designed to be reused across the application's lifetime.  
  * The service will handle authentication transparently by leveraging Google Cloud's default application credentials, which will be managed by .NET Aspire.  
  * The service will expose two primary, well-documented methods:

using Google.Cloud.AIPlatform.V1;  
using Microsoft.Extensions.Logging;  
using System;  
using System.Threading.Tasks;

public class GeminiService  
{  
    private readonly GenerativeModel \_geminiModel;  
    private readonly ILogger\<GeminiService\> \_logger;

    public GeminiService(ILogger\<GeminiService\> logger)  
    {  
        // The model name should be sourced from configuration for flexibility.  
        \_geminiModel \= new GenerativeModel("gemini-1.5-flash");  
        \_logger \= logger;  
    }

    /// \<summary\>  
    /// Generates a simple, benefit-oriented explanation for a given strategic concept.  
    /// This method uses a tightly-scoped prompt to ensure the LLM's response is focused and concise.  
    /// \</summary\>  
    public async Task\<string\> ExplainStrategyAsync(string strategyName, string problem, string solution)  
    {  
        var prompt \= $"""  
        You are an expert AI architect. Explain the primary benefit of the strategy named '{strategyName}' in one simple sentence, as if you were explaining it to a non-technical stakeholder.  
        The problem this strategy solves is: "{problem}".  
        The proposed solution is: "{solution}".  
        Focus only on the key business or technical advantage.  
        """;

        try  
        {  
            var response \= await \_geminiModel.GenerateContentAsync(prompt);  
            return response.Text;  
        }  
        catch (Exception ex)  
        {  
            \_logger.LogError(ex, "Error occurred while calling the Gemini API for strategy explanation.");  
            return "An error occurred while generating the explanation. Please try again later.";  
        }  
    }

    /// \<summary\>  
    /// Answers a user's question based on the overall context of the architectural report.  
    /// This method is grounded with a comprehensive context block to prevent the model from answering  
    /// outside the scope of the report, ensuring relevance and accuracy.  
    /// \</summary\>  
    public async Task\<string\> AnswerArbitraryQuestionAsync(string userQuestion)  
    {  
        var reportContext \= """  
        You are an expert AI architect and the author of a technical report. Your task is to answer a user's question based \*only\* on the context provided here.  
        The report compares a custom AI stack (GraphRAG for knowledge, Mem0 for memory) with a managed service (Google Vertex AI Memory Bank).  
        The report's main conclusion is that the custom stack offers superior control, transparency, and deep graph-based reasoning, making it ideal for complex, proprietary systems. The managed service is simpler and faster to implement but offers less control and transparency.  
        The report's primary recommendation is to enhance the custom stack with features inspired by the managed service, such as an automated 'memory curator' to clean and consolidate memories, and a 'living knowledge graph' that can learn from new conversations without requiring a full re-index.  
        """;

        var prompt \= $"{reportContext} Based on this, answer the following user question concisely and professionally: \\"{userQuestion}\\"";

        try  
        {  
            var response \= await \_geminiModel.GenerateContentAsync(prompt);  
            return response.Text;  
        }  
        catch (Exception ex)  
        {  
            \_logger.LogError(ex, "Error occurred while calling the Gemini API for Q\&A.");  
            return "An error occurred while processing your question. Please try again later.";  
        }  
    }  
}

### **2\. Updates to the ASP.NET Core API Service**

The main API service will be updated to include new endpoints for the AI-powered features and to integrate the new GeminiService.

* **Dependency Injection:** Register the GeminiService in MyAgent.ApiService/Program.cs. A Singleton lifetime is appropriate as the service is thread-safe and holds no per-request state.  
  // In Program.cs  
  builder.Services.AddSingleton\<GeminiService\>();

* **API Request Models:** Define strongly-typed request models for the new controller actions to ensure robust data validation.  
  // In a new file, e.g., Models/InteractiveFeatureRequests.cs  
  public record ExplainStrategyRequest(string StrategyName, string Problem, string Solution);  
  public record AskQuestionRequest(string Question);

* **New API Controller:** Create a new controller, InteractiveFeaturesController.cs, with clear OpenAPI/Swagger documentation.  
  // In InteractiveFeaturesController.cs  
  \[ApiController\]  
  \[Route("api/interactive")\]  
  public class InteractiveFeaturesController : ControllerBase  
  {  
      private readonly GeminiService \_geminiService;

      public InteractiveFeaturesController(GeminiService geminiService)  
      {  
          \_geminiService \= geminiService;  
      }

      /// \<summary\>  
      /// Generates a simplified explanation for a strategic recommendation.  
      /// \</summary\>  
      \[HttpPost("explain")\]  
      \[ProducesResponseType(typeof(object), 200)\]  
      public async Task\<IActionResult\> GetExplanation(\[FromBody\] ExplainStrategyRequest request)  
      {  
          var explanation \= await \_geminiService.ExplainStrategyAsync(request.StrategyName, request.Problem, request.Solution);  
          return Ok(new { explanation });  
      }

      /// \<summary\>  
      /// Answers a user's question about the architectural report.  
      /// \</summary\>  
      \[HttpPost("ask")\]  
      \[ProducesResponseType(typeof(object), 200)\]  
      public async Task\<IActionResult\> AskQuestion(\[FromBody\] AskQuestionRequest request)  
      {  
          var answer \= await \_geminiService.AnswerArbitraryQuestionAsync(request.Question);  
          return Ok(new { answer });  
      }  
  }

### **3\. Orchestration and Security with .NET Aspire**

The .NET Aspire AppHost remains the central point for configuration, ensuring that the ApiService has secure and reliable access to the necessary cloud credentials.

* **Secure Configuration:** Instead of hardcoding a file path, leverage Aspire's integration with .NET's configuration system. For local development, the path to the GOOGLE\_APPLICATION\_CREDENTIALS JSON file can be stored in secrets.json. For production, Aspire can be configured to pull this credential from a secure source like Azure Key Vault.  
  // In MyAgent.AppHost/Program.cs

  // For local development with user secrets  
  var apiService \= builder.AddProject\<Projects.MyAgent\_ApiService\>("apiservice")  
                          .WithEnvironment("GOOGLE\_APPLICATION\_CREDENTIALS", builder.Configuration\["GoogleCredentialsPath"\]);

  // For production deployment (e.g., to Azure)  
  // var keyVault \= builder.AddAzureKeyVault("mykeyvault");  
  // apiService.WithReference(keyVault);

### **4\. Impact on the Overall System**

* **Agent Logic:** The agent's core reasoning loop remains unchanged. These new features are designed for direct user interaction with the application's UI, not as tools for the agent itself. This maintains a clean separation between the agent's primary functions and the application's user-facing interactive elements. This design choice prevents the agent from becoming responsible for UI-specific tasks, adhering to a cleaner architectural pattern.  
* **Scalability and Resilience:** The GeminiService makes external network calls, which are potential points of failure and latency. To ensure the system is robust, standard resilience policies should be applied. The HttpClient registration for the Gemini client can be enhanced with policies from the **Polly** library to handle transient errors.  
  // Example of adding a retry policy in Program.cs  
  builder.Services.AddHttpClient\<GeminiService\>()  
      .AddTransientHttpErrorPolicy(policy \=\>   
          policy.WaitAndRetryAsync(3, retryAttempt \=\> TimeSpan.FromSeconds(Math.Pow(2, retryAttempt))));

  Additionally, for high-traffic scenarios, implementing a caching layer (e.g., using Redis, which is easily added to Aspire) for common Q\&A pairs can significantly reduce costs and improve response times.  
* **User Experience:** These features fundamentally change the nature of the application. The "Strategy Explainer" transforms static documentation into a dynamic, multi-layered educational tool, allowing users of varying technical depths to engage with the material. The "Interactive Q\&A" adds a layer of discoverability, empowering users to find information and clarify points without needing to read the entire report linearly. This transforms the application from a static report viewer into a dynamic and responsive knowledge tool.