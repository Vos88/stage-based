import React from 'react';
import { LatexRenderer } from './LatexRenderer';

export function LatexTest() {
  const testContent = `
    Here's some inline math: The slope is $\\beta_1 = \\frac{\\sum(x_i - \\bar{x})(y_i - \\bar{y})}{\\sum(x_i - \\bar{x})^2}$
    
    And here's a block equation:
    
    $$y = \\beta_0 + \\beta_1 x + \\varepsilon$$
    
    Another inline example: The probability is $P(y=1|x) = \\frac{1}{1 + e^{-(\\beta_0 + \\beta_1 x)}}$
    
    And a more complex block equation:
    
    $$\\text{RSS}(\\beta_0, \\beta_1) = \\sum_{i=1}^n \\left(y_i - \\beta_0 - \\beta_1 x_i\\right)^2$$
  `;

  return (
    <div className="p-8 max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold mb-6">LaTeX Rendering Test</h1>
      <div className="bg-card p-6 rounded-lg border">
        <LatexRenderer content={testContent} className="text-lg leading-relaxed" />
      </div>
    </div>
  );
}
