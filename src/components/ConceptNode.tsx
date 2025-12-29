import { ConceptNode as ConceptNodeType } from "@/data/aiConceptTree";

interface ConceptNodeProps {
  node: ConceptNodeType;
  onClick: (node: ConceptNodeType) => void;
}

// Map gradient colors to light background colors
const colorMap: Record<string, string> = {
  // Supervised Learning - Green family
  "bg-gradient-to-br from-green-500 to-emerald-600": "bg-green-50 border-green-200",
  "bg-gradient-to-br from-emerald-500 to-teal-600": "bg-emerald-50 border-emerald-200",
  "bg-gradient-to-br from-green-400 to-emerald-500": "bg-green-50 border-green-200",
  "bg-gradient-to-br from-lime-400 to-green-500": "bg-lime-50 border-lime-200",
  "bg-gradient-to-br from-teal-400 to-cyan-500": "bg-teal-50 border-teal-200",
  "bg-gradient-to-br from-emerald-400 to-teal-500": "bg-emerald-50 border-emerald-200",
  "bg-gradient-to-br from-green-500 to-lime-600": "bg-green-50 border-green-200",
  
  // Classification - Purple family
  "bg-gradient-to-br from-violet-500 to-purple-600": "bg-violet-50 border-violet-200",
  "bg-gradient-to-br from-purple-400 to-violet-500": "bg-purple-50 border-purple-200",
  "bg-gradient-to-br from-indigo-400 to-purple-500": "bg-indigo-50 border-indigo-200",
  "bg-gradient-to-br from-fuchsia-400 to-purple-500": "bg-fuchsia-50 border-fuchsia-200",
  "bg-gradient-to-br from-violet-400 to-purple-600": "bg-violet-50 border-violet-200",
  "bg-gradient-to-br from-purple-500 to-violet-600": "bg-purple-50 border-purple-200",
  
  // Unsupervised Learning - Cyan family
  "bg-gradient-to-br from-cyan-500 to-teal-600": "bg-cyan-50 border-cyan-200",
  "bg-gradient-to-br from-cyan-500 to-sky-600": "bg-cyan-50 border-cyan-200",
  "bg-gradient-to-br from-sky-500 to-blue-600": "bg-sky-50 border-sky-200",
  "bg-gradient-to-br from-cyan-400 to-sky-500": "bg-cyan-50 border-cyan-200",
  "bg-gradient-to-br from-sky-400 to-cyan-500": "bg-sky-50 border-sky-200",
  "bg-gradient-to-br from-blue-400 to-cyan-500": "bg-blue-50 border-blue-200",
  "bg-gradient-to-br from-sky-400 to-blue-500": "bg-sky-50 border-sky-200",
  "bg-gradient-to-br from-cyan-400 to-blue-500": "bg-cyan-50 border-cyan-200",
  
  // Reinforcement Learning - Orange family
  "bg-gradient-to-br from-orange-500 to-amber-600": "bg-orange-50 border-orange-200",
  "bg-gradient-to-br from-orange-400 to-yellow-500": "bg-orange-50 border-orange-200",
  "bg-gradient-to-br from-orange-500 to-red-500": "bg-orange-50 border-orange-200",
  "bg-gradient-to-br from-yellow-500 to-orange-600": "bg-yellow-50 border-yellow-200",
  
  // Neural Networks - Indigo family
  "bg-gradient-to-br from-indigo-500 to-blue-600": "bg-indigo-50 border-indigo-200",
  "bg-gradient-to-br from-indigo-400 to-blue-500": "bg-indigo-50 border-indigo-200",
  "bg-gradient-to-br from-blue-400 to-indigo-500": "bg-blue-50 border-blue-200",
  "bg-gradient-to-br from-blue-500 to-indigo-600": "bg-blue-50 border-blue-200",
  "bg-gradient-to-br from-indigo-500 to-violet-600": "bg-indigo-50 border-indigo-200",
  "bg-gradient-to-br from-violet-400 to-indigo-500": "bg-violet-50 border-violet-200",
  "bg-gradient-to-br from-indigo-400 to-violet-500": "bg-indigo-50 border-indigo-200",
  "bg-gradient-to-br from-blue-400 to-indigo-600": "bg-blue-50 border-blue-200",
  
  // Ensemble Learning - Red family
  "bg-gradient-to-br from-red-500 to-rose-600": "bg-red-50 border-red-200",
  "bg-gradient-to-br from-red-400 to-rose-500": "bg-red-50 border-red-200",
  "bg-gradient-to-br from-orange-500 to-red-600": "bg-orange-50 border-orange-200",
  "bg-gradient-to-br from-red-500 to-orange-600": "bg-red-50 border-red-200",
  "bg-gradient-to-br from-pink-400 to-rose-500": "bg-pink-50 border-pink-200",
  "bg-gradient-to-br from-red-400 to-orange-500": "bg-red-50 border-red-200",
  
  // Root and other
  "bg-gradient-to-br from-purple-600 to-blue-600": "bg-purple-50 border-purple-200",
  "bg-gradient-to-br from-blue-500 to-cyan-600": "bg-blue-50 border-blue-200",
  
  // Symbolic AI
  "bg-gradient-to-br from-indigo-500 to-purple-600": "bg-indigo-50 border-indigo-200",
};

export function ConceptNode({ node, onClick }: ConceptNodeProps) {
  const handleClick = () => {
    onClick(node);
  };

  // Get light color variant or default
  const lightColor = colorMap[node.color] || "bg-gray-50 border-gray-200";

  return (
    <div
      className={`concept-node fade-in ${lightColor}`}
      onClick={handleClick}
    >
      <div className="text-center space-y-2">
        <h3 className="text-xl font-semibold text-foreground">
          {node.title}
        </h3>
        <p className="text-sm text-muted-foreground leading-relaxed">
          {node.description}
        </p>
        {node.children && node.children.length > 0 && (
          <div className="mt-4 text-xs text-muted-foreground flex items-center justify-center gap-1">
            <span>Explore</span>
            <span className="text-primary">→</span>
          </div>
        )}
      </div>
    </div>
  );
}