
import { Link } from "react-router-dom";
import { Globe, BookOpen, Users } from "lucide-react";
import Navbar from "@/components/layout/Navbar";
import Footer from "@/components/layout/Footer";

const About = () => {
  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      
      <main className="flex-1">
        {/* Hero Section */}
        <section className="gradient-bg text-white py-16 md:py-24">
          <div className="container mx-auto px-4">
            <div className="max-w-3xl mx-auto text-center">
              <h1 className="text-4xl md:text-5xl font-bold mb-6">
                About TheDrawCode
              </h1>
              <p className="text-xl">
                Revolutionizing soccer predictions through advanced machine learning and data analytics.
              </p>
            </div>
          </div>
        </section>

        {/* Mission Section */}
        <section className="py-16 bg-white">
          <div className="container mx-auto px-4">
            <div className="max-w-3xl mx-auto text-center">
              <h2 className="text-3xl font-bold mb-6">Our Mission</h2>
              <p className="text-gray-600 text-lg">
                At TheDrawCode, we're dedicated to bringing transparency and accuracy to soccer predictions. 
                By combining cutting-edge machine learning algorithms with comprehensive historical data, 
                we provide insights that help our users make more informed decisions.
              </p>
            </div>
          </div>
        </section>

        {/* Features Section */}
        <section className="py-16 bg-gray-50">
          <div className="container mx-auto px-4">
            <h2 className="text-3xl font-bold text-center mb-12">What Sets Us Apart</h2>
            <div className="grid md:grid-cols-3 gap-8">
              <div className="bg-white p-6 rounded-lg shadow-sm">
                <Globe className="h-12 w-12 text-soccer-blue mb-4" />
                <h3 className="text-xl font-bold mb-3">Global Coverage</h3>
                <p className="text-gray-600">
                  Comprehensive analysis of matches from major leagues worldwide, providing insights for games across different continents.
                </p>
              </div>
              <div className="bg-white p-6 rounded-lg shadow-sm">
                <BookOpen className="h-12 w-12 text-soccer-blue mb-4" />
                <h3 className="text-xl font-bold mb-3">Deep Analysis</h3>
                <p className="text-gray-600">
                  Our AI models analyze thousands of data points including team performance, player statistics, and historical matchups.
                </p>
              </div>
              <div className="bg-white p-6 rounded-lg shadow-sm">
                <Users className="h-12 w-12 text-soccer-blue mb-4" />
                <h3 className="text-xl font-bold mb-3">Community Driven</h3>
                <p className="text-gray-600">
                  Join a growing community of sports enthusiasts and data analysts sharing insights and strategies.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* CTA Section */}
        <section className="py-16 bg-soccer-blue text-white">
          <div className="container mx-auto px-4 text-center">
            <h2 className="text-3xl font-bold mb-6">Ready to Get Started?</h2>
            <p className="text-xl mb-8 max-w-2xl mx-auto">
              Join thousands of users who trust TheDrawCode for accurate soccer predictions.
            </p>
            <Link 
              to="/matches" 
              className="inline-flex items-center justify-center px-8 py-3 border border-transparent text-base font-medium rounded-md text-soccer-blue bg-white hover:bg-gray-100 md:text-lg"
            >
              View Matches
            </Link>
          </div>
        </section>
      </main>
      
      <Footer />
    </div>
  );
};

export default About;
