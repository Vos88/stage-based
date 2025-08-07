import { useState, useCallback } from "react";
import { ConceptNode as ConceptNodeType } from "@/data/aiConceptTree";
import { ConceptNode } from "./ConceptNode";
import { ContentPanel } from "./ContentPanel";
import { BreadcrumbNav } from "./BreadcrumbNav";

interface ConceptExplorerProps {
  rootNode: ConceptNodeType;
}

export function ConceptExplorer({ rootNode }: ConceptExplorerProps) {
  const [navigationPath, setNavigationPath] = useState<ConceptNodeType[]>([rootNode]);
  const [showingContent, setShowingContent] = useState(false);

  const currentNode = navigationPath[navigationPath.length - 1];
  const currentChildren = currentNode.children || [];

  const handleNodeClick = useCallback((node: ConceptNodeType) => {
    if (node.children && node.children.length > 0) {
      // Navigate to children
      setNavigationPath(prev => [...prev, node]);
      setShowingContent(false);
    } else {
      // Show content for leaf node
      setNavigationPath(prev => [...prev, node]);
      setShowingContent(true);
    }
  }, []);

  const handleBreadcrumbNavigate = useCallback((nodeIndex: number) => {
    setNavigationPath(prev => prev.slice(0, nodeIndex + 1));
    setShowingContent(false);
  }, []);

  const handleBackFromContent = useCallback(() => {
    setNavigationPath(prev => prev.slice(0, -1));
    setShowingContent(false);
  }, []);

  return (
    <div className="min-h-screen bg-background">
      <div className="stage-container">
        <BreadcrumbNav 
          path={navigationPath} 
          onNavigate={handleBreadcrumbNavigate} 
        />

        {showingContent ? (
          <ContentPanel 
            node={currentNode} 
            onBack={handleBackFromContent} 
          />
        ) : (
          <div className="w-full max-w-6xl">
            {navigationPath.length === 1 ? (
              // Root level - show the main AI node
              <div className="flex justify-center">
                <div className="max-w-md">
                  <ConceptNode 
                    node={rootNode} 
                    onClick={handleNodeClick} 
                  />
                </div>
              </div>
            ) : (
              // Child levels - show grid of children
              <>
                <div className="text-center mb-8">
                  <h1 className="text-4xl font-bold text-foreground mb-2">
                    {currentNode.title}
                  </h1>
                  <p className="text-xl text-muted-foreground">
                    {currentNode.description}
                  </p>
                </div>
                <div className="node-grid">
                  {currentChildren.map((child) => (
                    <ConceptNode 
                      key={child.id} 
                      node={child} 
                      onClick={handleNodeClick} 
                    />
                  ))}
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}