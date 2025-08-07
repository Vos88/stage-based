import { ConceptNode as ConceptNodeType } from "@/data/aiConceptTree";

interface ConceptNodeProps {
  node: ConceptNodeType;
  onClick: (node: ConceptNodeType) => void;
}

export function ConceptNode({ node, onClick }: ConceptNodeProps) {
  const handleClick = () => {
    onClick(node);
  };

  return (
    <div
      className="concept-node fade-in"
      style={{ "--node-color": node.color } as React.CSSProperties}
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