import { useState, useRef, useEffect } from "react";

export default function App() {
  const [messages, setMessages] = useState([
    {
      sender: "bot",
      name: "Legal AI Assistant",
      text: "Hello! I am your Legal AI Assistant. How can I help you today?",
    },
  ]);

  const [input, setInput] = useState("");
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = () => {
    if (!input.trim()) return;

    setMessages((prev) => [
      ...prev,
      { sender: "user", name: "You", text: input },
      {
        sender: "bot",
        name: "Legal AI Assistant",
        text: "This is a dummy response.",
      },
    ]);

    setInput("");
  };

  return (
    <div className="h-screen flex bg-gradient-to-br from-slate-100 to-slate-200">

      {/* Sidebar */}
      <div className="w-64 bg-white border-r flex flex-col p-4">
        <h1 className="text-lg font-semibold mb-6">⚖️ Legal AI</h1>

        <button className="bg-indigo-500 text-white rounded-lg py-2 mb-4">
          + New Chat
        </button>

        <p className="text-sm text-gray-400">Today</p>

        <div className="bg-indigo-50 text-indigo-600 p-2 rounded-lg mt-2">
          New Conversation
        </div>

        <div className="mt-auto pt-4 border-t">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-full bg-indigo-500 text-white flex items-center justify-center">
              B
            </div>
            <div>
              <p className="text-sm font-medium">BiTbYter</p>
              <p className="text-xs text-gray-400">bpandey@123.com</p>
            </div>
          </div>
        </div>
      </div>

      {/* Main */}
      <div className="flex-1 flex flex-col">

        {/* Header */}
        <div className="px-6 py-4 bg-white border-b">
          <h2 className="text-lg font-semibold">Legal AI Assistant</h2>
          <p className="text-sm text-gray-500">Ask anything about law</p>
        </div>

        {/* Chat */}
        <div className="flex-1 overflow-y-auto px-16 py-8 space-y-8">

       {messages.map((msg, i) => (
  <div key={i} className="flex flex-col gap-1">

    {/* Name + Avatar */}
    <div
      className={`flex items-center gap-2 ${
        msg.sender === "user" ? "justify-end" : "justify-start"
      }`}
    >
      {msg.sender === "bot" && (
        <>
          <div className="w-8 h-8 rounded-full bg-indigo-500 text-white flex items-center justify-center text-sm shadow">
            🤖
          </div>
          <span className="text-sm font-medium text-indigo-600">
            {msg.name}
          </span>
        </>
      )}

      {msg.sender === "user" && (
        <>
          <span className="text-sm font-medium text-indigo-600">
            {msg.name}
          </span>
          <div className="w-8 h-8 rounded-full bg-gray-300 flex items-center justify-center text-sm shadow">
            B
          </div>
        </>
      )}
    </div>

    {/* Message Bubble */}
    <div
      className={`max-w-md px-4 py-3 rounded-2xl shadow-[0_4px_20px_rgba(0,0,0,0.08)] ${
        msg.sender === "user"
          ? "ml-auto bg-indigo-500 text-white"
          : "bg-white"
      }`}
    >
      {msg.text}
    </div>
  </div>
))}

          <div ref={bottomRef}></div>
        </div>

        {/* Input */}
       <div className="p-6 bg-transparent">
  <div className="max-w-2xl mx-auto bg-white shadow-[0_8px_30px_rgba(0,0,0,0.08)] rounded-full flex items-center px-4 py-2">

    <input
      className="flex-1 px-3 py-2 outline-none rounded-full"
      placeholder="Ask your legal question..."
      value={input}
      onChange={(e) => setInput(e.target.value)}
      onKeyDown={(e) => e.key === "Enter" && sendMessage()}
    />

    <button
      onClick={sendMessage}
      className="bg-indigo-500 hover:bg-indigo-600 transition text-white px-4 py-2 rounded-full"
    >
      ➤
    </button>
  </div>
</div>

      </div>
    </div>
  );
}