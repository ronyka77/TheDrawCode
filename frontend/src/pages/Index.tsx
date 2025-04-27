
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { useQuery } from "@tanstack/react-query";
import { API } from "@/services/api";
import MatchCard from "@/components/matches/MatchCard";
import Navbar from "@/components/layout/Navbar";
import Footer from "@/components/layout/Footer";

const Index = () => {
  const { data: upcomingMatches, isLoading } = useQuery({
    queryKey: ["upcomingMatches"],
    queryFn: API.matches.getUpcomingMatches,
  });

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      
      <main className="flex-1">
        {/* Hero Section */}
        <section className="gradient-bg text-white">
          <div className="container mx-auto px-4 py-16 md:py-24">
            <div className="max-w-3xl mx-auto text-center">
              <h1 className="text-4xl md:text-5xl font-bold mb-6">
                Predict Soccer Match Outcomes with Machine Learning
              </h1>
              <p className="text-xl mb-8">
                Get accurate predictions for upcoming matches based on advanced statistical models and machine learning algorithms.
              </p>
              <div className="space-x-4">
                <Button
                  asChild
                  size="lg"
                  className="bg-soccer-green hover:bg-soccer-green/90"
                >
                  <Link to="/matches">View Matches</Link>
                </Button>
                <Button
                  asChild
                  variant="outline"
                  size="lg"
                  className="text-white border-white hover:bg-white/10"
                >
                  <Link to="/about">Learn More</Link>
                </Button>
              </div>
            </div>
          </div>
        </section>
        
        {/* Features Section */}
        <section className="py-16 bg-white">
          <div className="container mx-auto px-4">
            <h2 className="text-3xl font-bold text-center mb-12">How It Works</h2>
            <div className="grid md:grid-cols-3 gap-8">
              <div className="text-center">
                <div className="bg-soccer-blue/10 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-soccer-blue" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                <h3 className="text-xl font-bold mb-2">Advanced Analytics</h3>
                <p className="text-gray-600">
                  Our AI analyzes thousands of data points from historical matches, team performance, player statistics, and more.
                </p>
              </div>
              <div className="text-center">
                <div className="bg-soccer-blue/10 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-soccer-blue" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                </div>
                <h3 className="text-xl font-bold mb-2">Real-time Updates</h3>
                <p className="text-gray-600">
                  Get real-time predictions and updates as new data becomes available, ensuring you always have the latest insights.
                </p>
              </div>
              <div className="text-center">
                <div className="bg-soccer-blue/10 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-soccer-blue" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                </div>
                <h3 className="text-xl font-bold mb-2">Performance Tracking</h3>
                <p className="text-gray-600">
                  Track the accuracy of predictions over time and gain insights into which leagues and teams our model performs best with.
                </p>
              </div>
            </div>
          </div>
        </section>
        
        {/* Upcoming Matches Section */}
        <section className="py-16 bg-gray-50">
          <div className="container mx-auto px-4">
            <div className="flex justify-between items-center mb-8">
              <h2 className="text-2xl font-bold">Upcoming Matches</h2>
              <Link to="/matches" className="text-soccer-green hover:underline font-medium">
                View all matches →
              </Link>
            </div>
            
            {isLoading ? (
              <div className="text-center py-8">
                <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-soccer-green border-t-transparent"></div>
                <p className="mt-2 text-gray-600">Loading matches...</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {upcomingMatches?.slice(0, 3).map((match) => (
                  <MatchCard key={match.id} match={match} />
                ))}
              </div>
            )}
          </div>
        </section>
        
        {/* CTA Section */}
        <section className="py-16 bg-soccer-blue text-white">
          <div className="container mx-auto px-4 text-center">
            <h2 className="text-3xl font-bold mb-6">Ready to Start Predicting?</h2>
            <p className="text-xl mb-8 max-w-2xl mx-auto">
              Join thousands of users who use our AI-powered predictions for better insights into soccer matches.
            </p>
            <Button
              size="lg"
              className="bg-soccer-green hover:bg-soccer-green/90"
              asChild
            >
              <Link to="/matches">Get Started for Free</Link>
            </Button>
          </div>
        </section>
      </main>
      
      <Footer />
    </div>
  );
};

export default Index;
