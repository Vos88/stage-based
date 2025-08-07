import { ConceptNode } from "@/data/aiConceptTree";
import { Button } from "@/components/ui/button";

interface ContentPanelProps {
  node: ConceptNode;
  onBack: () => void;
}

export function ContentPanel({ node, onBack }: ContentPanelProps) {
  return (
    <div className="content-card fade-in max-w-4xl w-full">
      <div className="flex items-start justify-between mb-6">
        <div>
          <h1 className="text-3xl font-bold text-foreground mb-2">
            {node.title}
          </h1>
          <p className="text-lg text-muted-foreground">
            {node.description}
          </p>
        </div>
        <Button variant="outline" onClick={onBack}>
          ← Back
        </Button>
      </div>

      {node.content && (
        <div className="space-y-6">
          {node.content.keyPoints && (
            <div>
              <h3 className="text-xl font-semibold mb-3">Key Points</h3>
              <ul className="space-y-2">
                {node.content.keyPoints.map((point, index) => (
                  <li key={index} className="flex items-start gap-2">
                    <span className="text-primary mt-1">•</span>
                    <span className="text-muted-foreground">{point}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {node.content.codeExample && (
            <div>
              <h3 className="text-xl font-semibold mb-3">Code Example</h3>
              <div className="code-block">
                <pre className="text-sm">
                  <code>{node.content.codeExample}</code>
                </pre>
              </div>
            </div>
          )}

          {node.content.links && node.content.links.length > 0 && (
            <div>
              <h3 className="text-xl font-semibold mb-3">Additional Resources</h3>
              <div className="space-y-2">
                {node.content.links.map((link, index) => (
                  <a
                    key={index}
                    href={link.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-2 text-primary hover:text-primary/80 transition-colors"
                  >
                    <span>{link.title}</span>
                    <span className="text-xs">↗</span>
                  </a>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}