import Navbar from "@/components/layout/Navbar";
import Footer from "@/components/layout/Footer";

const GDPR = () => {
  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      
      <main className="flex-1 py-8">
        <div className="container mx-auto px-4 prose prose-invert max-w-4xl">
          <h1 className="text-4xl font-bold mb-8">GDPR Compliance</h1>
          
          <section className="mb-8">
            <h2 className="text-2xl font-semibold mb-4">Your Rights Under GDPR</h2>
            <ul className="list-disc pl-6 space-y-2">
              <li>Right to access your personal data</li>
              <li>Right to rectification of inaccurate personal data</li>
              <li>Right to erasure ("right to be forgotten")</li>
              <li>Right to restrict processing</li>
              <li>Right to data portability</li>
              <li>Right to object to processing</li>
            </ul>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-semibold mb-4">Data Processing</h2>
            <p>
              We process your data according to GDPR principles: lawfulness, fairness, and transparency; 
              purpose limitation; data minimization; accuracy; storage limitation; integrity and confidentiality.
            </p>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-semibold mb-4">International Transfers</h2>
            <p>
              When we transfer your personal data outside the EEA, we ensure appropriate safeguards are in place 
              through standard contractual clauses or other legal mechanisms.
            </p>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-semibold mb-4">Data Protection Officer</h2>
            <p>
              For any GDPR-related inquiries, you can contact our Data Protection Officer at dpo@thedrawcode.com
            </p>
          </section>

          <p className="text-sm text-gray-400 mt-8">
            Last updated: {new Date().toLocaleDateString()}
          </p>
        </div>
      </main>
      
      <Footer />
    </div>
  );
};

export default GDPR;
