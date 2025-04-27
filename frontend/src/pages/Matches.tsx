
import Navbar from "@/components/layout/Navbar";
import Footer from "@/components/layout/Footer";
import MatchesList from "@/components/matches/MatchesList";

const Matches = () => {
  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      
      <main className="flex-1 py-8">
        <div className="container mx-auto px-4">
          <MatchesList />
        </div>
      </main>
      
      <Footer />
    </div>
  );
};

export default Matches;
