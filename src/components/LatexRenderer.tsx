import React from 'react';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';

interface LatexRendererProps {
  content: string;
  className?: string;
}

export function LatexRenderer({ content, className = "" }: LatexRendererProps) {
  // Function to split content into text and math parts
  const parseContent = (text: string) => {
    const parts: Array<{ type: 'text' | 'inline-math' | 'block-math'; content: string }> = [];
    
    // Match block math: $$...$$
    const blockMathRegex = /\$\$([\s\S]*?)\$\$/g;
    let lastIndex = 0;
    let match;
    
    while ((match = blockMathRegex.exec(text)) !== null) {
      // Add text before the math
      if (match.index > lastIndex) {
        parts.push({
          type: 'text',
          content: text.slice(lastIndex, match.index)
        });
      }
      
      // Add the block math
      parts.push({
        type: 'block-math',
        content: match[1].trim()
      });
      
      lastIndex = match.index + match[0].length;
    }
    
    // Add remaining text
    if (lastIndex < text.length) {
      parts.push({
        type: 'text',
        content: text.slice(lastIndex)
      });
    }
    
    // Process each part for inline math
    const finalParts: Array<{ type: 'text' | 'inline-math' | 'block-math'; content: string }> = [];
    
    parts.forEach(part => {
      if (part.type === 'block-math') {
        finalParts.push(part);
      } else {
        // Process text for inline math: $...$
        const inlineMathRegex = /\$([^\$]+)\$/g;
        let lastIndex = 0;
        let match;
        
        while ((match = inlineMathRegex.exec(part.content)) !== null) {
          // Add text before the math
          if (match.index > lastIndex) {
            finalParts.push({
              type: 'text',
              content: part.content.slice(lastIndex, match.index)
            });
          }
          
          // Add the inline math
          finalParts.push({
            type: 'inline-math',
            content: match[1].trim()
          });
          
          lastIndex = match.index + match[0].length;
        }
        
        // Add remaining text
        if (lastIndex < part.content.length) {
          finalParts.push({
            type: 'text',
            content: part.content.slice(lastIndex)
          });
        }
      }
    });
    
    return finalParts;
  };

  const parts = parseContent(content);

  return (
    <div className={className}>
      {parts.map((part, index) => {
        if (part.type === 'text') {
          return <span key={index}>{part.content}</span>;
        } else if (part.type === 'inline-math') {
          try {
            return <InlineMath key={index} math={part.content} />;
          } catch (error) {
            console.error('Invalid inline math:', part.content, error);
            return <span key={index} className="text-red-500">${part.content}$</span>;
          }
        } else if (part.type === 'block-math') {
          try {
            return (
              <div key={index} className="my-4 flex justify-center">
                <BlockMath math={part.content} />
              </div>
            );
          } catch (error) {
            console.error('Invalid block math:', part.content, error);
            return (
              <div key={index} className="my-4 text-red-500 text-center">
                $${part.content}$$
              </div>
            );
          }
        }
        return null;
      })}
    </div>
  );
}
