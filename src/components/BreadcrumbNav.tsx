import { ConceptNode } from "@/data/aiConceptTree";

interface BreadcrumbNavProps {
  path: ConceptNode[];
  onNavigate: (nodeIndex: number) => void;
}

export function BreadcrumbNav({ path, onNavigate }: BreadcrumbNavProps) {
  if (path.length === 0) return null;

  return (
    <nav className="breadcrumb-nav">
      {path.map((node, index) => (
        <div key={node.id} className="flex items-center gap-2">
          <button
            onClick={() => onNavigate(index)}
            className="breadcrumb-item"
          >
            {node.title}
          </button>
          {index < path.length - 1 && (
            <span className="breadcrumb-separator">→</span>
          )}
        </div>
      ))}
    </nav>
  );
}